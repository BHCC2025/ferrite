//! The per-request protections as every engine applies them: one [`PhaseGuard`] per request around the
//! [`PhaseTracker`] (reasoning cap, loop breaker, repeated-sentence rule), fed with the tokens the client gets, one at
//! a time and in order, ACCEPTED tokens only (drafts that were verified, the engine's own forced token included).
//! The engine acts at exactly the token where the guard answers and gives back the tokens a verify accepted past
//! it, so what a protection does never depends on how many rows a step verified, and a request whose guard never
//! acts gets the tokens it would get without one.
//!
//! The guard also carries the sampling of each phase (reasoning block / answer / boosted stretch after an
//! answer-phase loop) in the engine's own type `S` and reports a change at the token where the phase changes.
//! The 27B scheduler wraps it as its `fe_model::sched::Guard` (ferrite.rs), the Flash-Next engine loop calls it
//! directly (fnext.rs).
use std::sync::{Arc, Mutex};

use crate::engine::SamplingParams;
use crate::loopdet::{PhaseEvents, PhaseTracker, Trig};

/// What the guard asks for after an accepted token.
#[derive(Clone, Debug, PartialEq)]
pub enum Act<S> {
    None,
    /// the engine writes this token itself as the next one (the end of the reasoning block); the model goes on after it
    Force(u32),
    /// end the request here (finish reason length)
    End,
    /// the tokens after this one are drawn with this sampling (reasoning -> answer, a boosted stretch and its end)
    Sampling(S),
    /// rewind the sequence by the first number of accepted tokens (this one included), of which the second number were held back by the
    /// engine while `holding()` and are dropped; draw the next tokens with this sampling
    Rewind(usize, usize, S),
}

/// The request's sampling parameters in its three phases: reasoning block, answer, boosted stretch. The reasoning
/// block's own values (`SamplingParams::reasoning`) replace the seven sampling fields only; the limits, the stop
/// tokens and the seed are the request's. A greedy answer loops deterministically, so its boosted stretch samples
/// (0.7 / 0.95); a sampled answer keeps its own sampling there.
pub fn phase_params(p: &SamplingParams) -> [SamplingParams; 4] {
    let answer = p.clone();
    let reasoning = match p.reasoning.as_deref().filter(|_| p.think_end.is_some()) {
        Some(r) => SamplingParams {
            temperature: r.temperature,
            top_p: r.top_p,
            top_k: r.top_k,
            min_p: r.min_p,
            presence_penalty: r.presence_penalty,
            frequency_penalty: r.frequency_penalty,
            repetition_penalty: r.repetition_penalty,
            ..p.clone()
        },
        None => answer.clone(),
    };
    let boost = if answer.temperature <= 0.0 { SamplingParams { temperature: 0.7, top_p: 0.95, ..answer.clone() } } else { answer.clone() };
    // the stretch after a tool-call rewind: the profile's sampling for a greedy request
    let recover = if answer.temperature <= 0.0 { SamplingParams { temperature: p.loop_rules.recover_temperature, top_p: p.loop_rules.recover_top_p, ..answer.clone() } } else { answer.clone() };
    [reasoning, answer, boost, recover]
}

/// The protections of one request. The tracker is shared (the 27B adapter reads what fired from its sink).
pub struct PhaseGuard<S> {
    tr: Arc<Mutex<PhaseTracker>>,
    think_end: Option<u32>,
    /// reasoning, answer, boosted stretch
    modes: [S; 4],
    mode: usize,
}

impl<S: Clone + PartialEq> PhaseGuard<S> {
    /// The request's guard and the sampling it starts with; None when it has no reasoning block and no loop breaker.
    /// `map` turns a phase's parameters into the engine's sampling type.
    pub fn new(p: &SamplingParams, map: impl Fn(&SamplingParams) -> S) -> Option<(Self, S)> {
        if p.think_end.is_none() && !p.loop_breaker {
            return None;
        }
        let modes = phase_params(p).map(|q| map(&q));
        let mode = if p.think_end.is_some() { 0 } else { 1 };
        let start = modes[mode].clone();
        Some((Self { tr: Arc::new(Mutex::new(PhaseTracker::new(p))), think_end: p.think_end, modes, mode }, start))
    }

    pub fn tracker(&self) -> Arc<Mutex<PhaseTracker>> {
        self.tr.clone()
    }

    /// What fired so far.
    pub fn events(&self) -> PhaseEvents {
        self.tr.lock().unwrap_or_else(|e| e.into_inner()).events.clone()
    }

    /// Tokens accepted from now on are held back from the client (see `Act::Rewind`).
    pub fn holding(&self) -> bool {
        self.tr.lock().unwrap_or_else(|e| e.into_inner()).holding()
    }

    /// The engine holds tokens back and can rewind: the tool-call recovery may be used.
    pub fn enable_recover(&self, force: bool) {
        self.tr.lock().unwrap_or_else(|e| e.into_inner()).enable_recover(force)
    }

    /// The answer phase's sampling.
    pub fn answer(&self) -> &S {
        &self.modes[1]
    }

    /// One accepted token. The cap and a reasoning cycle make the engine write the block's end token itself (the
    /// model then answers); an answer / tool-call cycle switches to the boosted sampling for a stretch and ends the
    /// request if it comes back.
    pub fn feed(&mut self, t: u32) -> Act<S> {
        let mut tr = self.tr.lock().unwrap_or_else(|e| e.into_inner());
        match tr.feed(&[t]) {
            Trig::EndLoop => return Act::End,
            Trig::Close => return self.think_end.map_or(Act::End, Act::Force),
            Trig::Rewind(n, held) => {
                self.mode = 3;
                return Act::Rewind(n, held, self.modes[3].clone());
            }
            Trig::None => {}
        }
        let mode = if tr.in_reasoning() { 0 } else if tr.boosting() { if tr.recovering() { 3 } else { 2 } } else { 1 };
        if mode != self.mode {
            let changed = self.modes[mode] != self.modes[self.mode];
            self.mode = mode;
            if changed {
                return Act::Sampling(self.modes[mode].clone());
            }
        }
        Act::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TE: u32 = 900;

    /// The seven sampling fields (what an engine's own sampling type would hold).
    fn key(p: &SamplingParams) -> (f32, f32, usize, f32, f32, f32, f32) {
        (p.temperature, p.top_p, p.top_k, p.min_p, p.presence_penalty, p.frequency_penalty, p.repetition_penalty)
    }

    #[test]
    fn no_guard_without_a_reasoning_block_or_the_breaker() {
        let p = SamplingParams { temperature: 0.0, ..Default::default() };
        assert!(PhaseGuard::new(&p, key).is_none());
    }

    #[test]
    fn the_reasoning_values_replace_the_sampling_fields_only() {
        let mut p = SamplingParams { temperature: 0.0, think_end: Some(TE), max_tokens: 77, seed: Some(5), stop_token_ids: vec![3], ..Default::default() };
        let r = SamplingParams { temperature: 1.0, top_p: 0.95, top_k: 20, max_tokens: 1, seed: None, ..Default::default() };
        p.reasoning = Some(Box::new(r));
        let [reasoning, answer, boost, _] = phase_params(&p);
        assert_eq!(key(&reasoning), (1.0, 0.95, 20, 0.0, 0.0, 0.0, 1.0));
        assert!(reasoning.max_tokens == 77 && reasoning.seed == Some(5) && reasoning.stop_token_ids == vec![3]);
        assert_eq!(answer, p);
        assert_eq!((boost.temperature, boost.top_p), (0.7, 0.95));
        // a sampled answer keeps its own sampling in the boosted stretch; no reasoning block: the reasoning values are not used
        let mut q = SamplingParams { temperature: 0.6, loop_breaker: true, ..Default::default() };
        q.reasoning = p.reasoning.clone();
        let [reasoning, answer, boost, _] = phase_params(&q);
        assert!(reasoning == answer && boost == answer);
    }

    #[test]
    fn the_cap_forces_the_block_end_and_the_sampling_switches_at_its_token() {
        let mut p = SamplingParams { temperature: 0.0, think_end: Some(TE), max_reasoning: 5, ..Default::default() };
        let mut r = p.clone();
        r.temperature = 1.0;
        r.think_end = None;
        p.reasoning = Some(Box::new(r));
        let (mut g, start) = PhaseGuard::new(&p, key).unwrap();
        assert_eq!(start.0, 1.0);
        let acts: Vec<Act<_>> = [1, 2, 3, 4, 5].iter().map(|&t| g.feed(t)).collect();
        assert!(acts[..4].iter().all(|a| *a == Act::None) && acts[4] == Act::Force(TE));
        match g.feed(TE) {
            Act::Sampling(s) => assert_eq!(s.0, 0.0),
            x => panic!("{x:?}"),
        }
        assert_eq!(g.feed(6), Act::None);
        let ev = g.events();
        assert!(ev.reasoning_capped && ev.reasoning_tokens == 6);
        // one sampling for both phases: the block's end changes nothing
        let p = SamplingParams { temperature: 0.0, think_end: Some(TE), ..Default::default() };
        let (mut g, _) = PhaseGuard::new(&p, key).unwrap();
        assert!([1, 2, TE, 3, 4].iter().all(|&t| g.feed(t) == Act::None));
    }
}
