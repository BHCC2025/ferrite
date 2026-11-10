//! Recorded command lists replayed on a queue: the CUDA-graph analogue for the per-token launch sequence (~1,000 launches a
//! decode step). A [`Graph`] is a regular, in-order Level Zero command list: kernel launches are recorded once (their arguments
//! and group counts are captured at record time), the list is closed, and every replay appends the whole list to an immediate
//! queue in one call (`zeCommandListImmediateAppendCommandListsExp`). With `mutable` (ZE_experimental_mutable_command_list) a
//! recorded launch keeps a command id whose scalar / pointer arguments and group count can be changed between replays
//! (`zeCommandListUpdateMutableCommandsExp`), e.g. the position and the attention chunk count of a decode step.
use crate::ffi::*;
use crate::{Kernel, KernelArg, Queue, Result, Ze, ZeError};
use std::ffi::c_void;
use std::sync::Arc;

pub struct Graph {
    ze: Arc<Ze>,
    list: ze_command_list_handle_t,
    mutable: bool,
    closed: bool,
    /// recorded launches
    pub launches: u32,
}
unsafe impl Send for Graph {}

fn check(r: ze_result_t, call: &'static str) -> Result<()> {
    if r == ZE_RESULT_SUCCESS { Ok(()) } else { Err(ZeError { code: r, call, log: None }) }
}

impl Graph {
    pub fn new(ze: &Arc<Ze>, mutable: bool) -> Result<Graph> {
        let md = ze_mutable_command_list_exp_desc_t { stype: ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC, pNext: std::ptr::null(), flags: 0 };
        let d = ze_command_list_desc_t {
            stype: ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
            pNext: if mutable { &md as *const _ as *const c_void } else { std::ptr::null() },
            commandQueueGroupOrdinal: 0,
            flags: ZE_COMMAND_LIST_FLAG_IN_ORDER,
        };
        let mut list = std::ptr::null_mut();
        check(unsafe { (ze.api.zeCommandListCreate)(ze.ctx, ze.device, &d, &mut list) }, "zeCommandListCreate")?;
        Ok(Graph { ze: ze.clone(), list, mutable, closed: false, launches: 0 })
    }

    /// Records a launch of `kernel` (its current arguments and group size) over `groups`. Mutable graphs return the command id.
    pub fn launch(&mut self, kernel: &Kernel, groups: [u32; 3]) -> Result<Option<u64>> {
        assert!(!self.closed, "graph already closed");
        let mut id = None;
        if self.mutable {
            let d = ze_mutable_command_id_exp_desc_t {
                stype: ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_ID_EXP_DESC,
                pNext: std::ptr::null(),
                flags: ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS | ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_COUNT,
            };
            let mut c = 0u64;
            check(unsafe { (self.ze.api.zeCommandListGetNextCommandIdExp)(self.list, &d, &mut c) }, "zeCommandListGetNextCommandIdExp")?;
            id = Some(c);
        }
        let g = ze_group_count_t { groupCountX: groups[0], groupCountY: groups[1], groupCountZ: groups[2] };
        check(unsafe { (self.ze.api.zeCommandListAppendLaunchKernel)(self.list, kernel.raw(), &g, std::ptr::null_mut(), 0, std::ptr::null_mut()) },
            "zeCommandListAppendLaunchKernel")?;
        self.launches += 1;
        Ok(id)
    }

    /// Records a device copy (USM pointers).
    pub fn copy(&mut self, dst: crate::DevPtr, src: crate::DevPtr, bytes: usize) -> Result<()> {
        assert!(!self.closed, "graph already closed");
        check(unsafe { (self.ze.api.zeCommandListAppendMemoryCopy)(self.list, dst.0 as *mut c_void, src.0 as *const c_void, bytes, std::ptr::null_mut(), 0, std::ptr::null_mut()) },
            "zeCommandListAppendMemoryCopy")
    }

    pub fn close(&mut self) -> Result<()> {
        if !self.closed {
            check(unsafe { (self.ze.api.zeCommandListClose)(self.list) }, "zeCommandListClose")?;
            self.closed = true;
        }
        Ok(())
    }

    /// Appends the whole recorded list to `q` (closes it first if needed). In-order with everything else on `q`, but the same list
    /// must not be appended again while a previous replay of it may still run (measured: two back-to-back replays of one list
    /// applied once): sync the queue (a decode step does, for its argmax) or use two graphs alternately.
    pub fn replay(&mut self, q: &Queue) -> Result<()> {
        self.close()?;
        let mut l = self.list;
        check(unsafe { (self.ze.api.zeCommandListImmediateAppendCommandListsExp)(q.raw(), 1, &mut l, std::ptr::null_mut(), 0, std::ptr::null_mut()) },
            "zeCommandListImmediateAppendCommandListsExp")
    }

    /// New values for arguments of recorded launches (mutable graphs): (command id, argument index, value). Waits for nothing:
    /// call it when no replay of this graph is still running.
    pub fn set_args(&mut self, updates: &[(u64, u32, &dyn KernelArg)]) -> Result<()> {
        assert!(self.mutable, "not a mutable graph");
        let mut descs: Vec<ze_mutable_kernel_argument_exp_desc_t> = updates
            .iter()
            .map(|(id, idx, v)| {
                let (n, p) = v.raw();
                ze_mutable_kernel_argument_exp_desc_t {
                    stype: ZE_STRUCTURE_TYPE_MUTABLE_KERNEL_ARGUMENT_EXP_DESC,
                    pNext: std::ptr::null(),
                    commandId: *id,
                    argIndex: *idx,
                    argSize: n,
                    pArgValue: p,
                }
            })
            .collect();
        for i in (1..descs.len()).rev() {
            let next = &descs[i] as *const _ as *const c_void;
            descs[i - 1].pNext = next;
        }
        self.update(if descs.is_empty() { std::ptr::null() } else { &descs[0] as *const _ as *const c_void })
    }

    /// New group counts for recorded launches (mutable graphs).
    pub fn set_groups(&mut self, updates: &[(u64, [u32; 3])]) -> Result<()> {
        assert!(self.mutable, "not a mutable graph");
        let counts: Vec<ze_group_count_t> =
            updates.iter().map(|(_, g)| ze_group_count_t { groupCountX: g[0], groupCountY: g[1], groupCountZ: g[2] }).collect();
        let mut descs: Vec<ze_mutable_group_count_exp_desc_t> = updates
            .iter()
            .zip(&counts)
            .map(|((id, _), c)| ze_mutable_group_count_exp_desc_t {
                stype: ZE_STRUCTURE_TYPE_MUTABLE_GROUP_COUNT_EXP_DESC,
                pNext: std::ptr::null(),
                commandId: *id,
                pGroupCount: c,
            })
            .collect();
        for i in (1..descs.len()).rev() {
            let next = &descs[i] as *const _ as *const c_void;
            descs[i - 1].pNext = next;
        }
        self.update(if descs.is_empty() { std::ptr::null() } else { &descs[0] as *const _ as *const c_void })
    }

    fn update(&mut self, chain: *const c_void) -> Result<()> {
        if chain.is_null() {
            return Ok(());
        }
        let d = ze_mutable_commands_exp_desc_t { stype: ZE_STRUCTURE_TYPE_MUTABLE_COMMANDS_EXP_DESC, pNext: chain, flags: 0 };
        check(unsafe { (self.ze.api.zeCommandListUpdateMutableCommandsExp)(self.list, &d) }, "zeCommandListUpdateMutableCommandsExp")?;
        // an updated mutable list must be closed again before it executes
        check(unsafe { (self.ze.api.zeCommandListClose)(self.list) }, "zeCommandListClose")?;
        self.closed = true;
        Ok(())
    }

    /// Empties the list for re-recording.
    pub fn reset(&mut self) -> Result<()> {
        check(unsafe { (self.ze.api.zeCommandListReset)(self.list) }, "zeCommandListReset")?;
        self.closed = false;
        self.launches = 0;
        Ok(())
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        unsafe { (self.ze.api.zeCommandListDestroy)(self.list) };
    }
}
