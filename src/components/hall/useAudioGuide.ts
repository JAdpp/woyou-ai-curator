"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  logEvent,
  prefetchAudioGuide,
  resolveAudioGuideUrl,
  type AudioGuideKind,
} from "@/lib/api";
import {
  useClientValue,
  useSpeechSynthesisSupported,
} from "@/lib/useClientCapability";

export interface AudioGuideSegment {
  kind: AudioGuideKind;
  ref?: string;
  /** Used only by the device-voice fallback; Qwen reads server-authored copy. */
  text: string;
}

export type AudioGuideStatus =
  | "preparing"
  | "playing"
  | "paused"
  | "fallback"
  | "error";

interface AudioGuideOptions {
  exhibitionId: string;
  segment: AudioGuideSegment;
  nextSegment: AudioGuideSegment | null;
  /** Camera movement and free-walk mode both pause narration. */
  active: boolean;
}

const FALLBACK_MESSAGE = "专业语音暂不可用，已使用设备语音";
const FALLBACK_ERROR_MESSAGE = "专业语音与设备语音均暂不可用，请稍后重试";

function guideSegmentKey(segment: AudioGuideSegment) {
  return `${segment.kind}:${segment.ref ?? ""}`;
}

/**
 * Qwen TTS-backed guide with browser speech as a visible resilience fallback.
 *
 * One HTMLAudioElement persists for the lifetime of the hall. Its URL is set
 * and play() is called synchronously from the visitor's click, so Safari keeps
 * the user activation while it waits for the server to generate the MP3. A
 * monotonically increasing request id prevents late media events from a
 * previous stop from affecting the current one.
 */
export function useAudioGuide({
  exhibitionId,
  segment,
  nextSegment,
  active,
}: AudioGuideOptions) {
  const [status, setStatus] = useState<AudioGuideStatus>("paused");
  const [message, setMessage] = useState<string | null>(null);
  const [fallbackSpeaking, setFallbackSpeaking] = useState(false);
  const statusRef = useRef<AudioGuideStatus>("paused");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const loadedSegmentRef = useRef<string | null>(null);
  const requestIdRef = useRef(0);
  const fallbackRequestRef = useRef<number | null>(null);
  const selectedRef = useRef(false);
  const activeRef = useRef(active);
  const segmentRef = useRef(segment);
  const nextSegmentRef = useRef(nextSegment);
  const previousSegmentKeyRef = useRef(guideSegmentKey(segment));
  const previousActiveRef = useRef(active);
  const voiceRef = useRef<SpeechSynthesisVoice | null>(null);
  const speechTimerRef = useRef<number | null>(null);
  const fallbackSpeakingRef = useRef(false);

  const speechSupported = useSpeechSynthesisSupported();
  const audioSupported = useClientValue(
    () => typeof window !== "undefined" && typeof window.Audio !== "undefined",
    false,
  );
  const supported = audioSupported || speechSupported;

  const updateStatus = useCallback((next: AudioGuideStatus) => {
    statusRef.current = next;
    setStatus(next);
  }, []);

  const updateFallbackSpeaking = useCallback((next: boolean) => {
    fallbackSpeakingRef.current = next;
    setFallbackSpeaking(next);
  }, []);

  const cancelSpeech = useCallback(() => {
    if (speechTimerRef.current !== null) {
      window.clearTimeout(speechTimerRef.current);
      speechTimerRef.current = null;
    }
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    updateFallbackSpeaking(false);
  }, [updateFallbackSpeaking]);

  const clearAudioSource = useCallback(() => {
    const audio = audioRef.current;
    if (audio) {
      audio.onplaying = null;
      audio.onended = null;
      audio.onerror = null;
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
    }
    loadedSegmentRef.current = null;
  }, []);

  const pauseAudio = useCallback(() => {
    audioRef.current?.pause();
    cancelSpeech();
    updateStatus("paused");
  }, [cancelSpeech, updateStatus]);

  const beginDeviceFallback = useCallback((
    requestId: number,
    target: AudioGuideSegment,
  ) => {
    if (
      requestIdRef.current !== requestId ||
      fallbackRequestRef.current === requestId ||
      !selectedRef.current ||
      !activeRef.current ||
      guideSegmentKey(segmentRef.current) !== guideSegmentKey(target)
    ) {
      return;
    }
    fallbackRequestRef.current = requestId;

    clearAudioSource();
    if (!speechSupported || !target.text.trim()) {
      selectedRef.current = false;
      updateStatus("error");
      setMessage(FALLBACK_ERROR_MESSAGE);
      return;
    }

    cancelSpeech();
    updateStatus("fallback");
    setMessage(FALLBACK_MESSAGE);

    const utterance = new SpeechSynthesisUtterance(target.text.slice(0, 900));
    if (voiceRef.current) utterance.voice = voiceRef.current;
    utterance.lang = voiceRef.current?.lang ?? "zh-CN";
    utterance.rate = 0.96;
    utterance.pitch = 1;
    utterance.onstart = () => {
      if (requestIdRef.current === requestId) updateFallbackSpeaking(true);
    };
    utterance.onend = () => {
      if (requestIdRef.current === requestId) updateFallbackSpeaking(false);
    };
    utterance.onerror = () => {
      if (requestIdRef.current !== requestId) return;
      selectedRef.current = false;
      updateFallbackSpeaking(false);
      updateStatus("error");
      setMessage(FALLBACK_ERROR_MESSAGE);
    };

    // Chromium may discard an utterance queued in the same tick as cancel().
    speechTimerRef.current = window.setTimeout(() => {
      speechTimerRef.current = null;
      if (requestIdRef.current === requestId && activeRef.current) {
        window.speechSynthesis.speak(utterance);
      }
    }, 120);
  }, [cancelSpeech, clearAudioSource, speechSupported, updateFallbackSpeaking, updateStatus]);

  const beginNarration = useCallback((target: AudioGuideSegment) => {
    if (!selectedRef.current || !activeRef.current || !target.text.trim()) return;

    const requestId = ++requestIdRef.current;
    fallbackRequestRef.current = null;
    cancelSpeech();
    setMessage(null);
    updateStatus("preparing");

    const audio = audioRef.current;
    if (!audio) {
      beginDeviceFallback(requestId, target);
      return;
    }

    clearAudioSource();
    loadedSegmentRef.current = guideSegmentKey(target);
    audio.src = resolveAudioGuideUrl(exhibitionId, target);
    audio.preload = "auto";
    audio.onplaying = () => {
      if (requestIdRef.current === requestId) updateStatus("playing");
    };
    audio.onended = () => {
      if (requestIdRef.current === requestId) updateStatus("paused");
    };
    audio.onerror = () => beginDeviceFallback(requestId, target);

    // Keep this call in the same stack as toggle() for Safari user activation.
    const playAttempt = audio.play();
    void playAttempt
      .then(() => {
        if (requestIdRef.current === requestId) updateStatus("playing");
      })
      .catch((error: unknown) => {
        if (requestIdRef.current !== requestId) return;
        if (error instanceof DOMException && error.name === "AbortError") return;
        if (error instanceof DOMException && error.name === "NotAllowedError") {
          updateStatus("paused");
          setMessage("请点击“彦远专业讲述”开始播放");
          return;
        }
        beginDeviceFallback(requestId, target);
      });

    // Starts after play() and never delays the current stop.
    prefetchAudioGuide(exhibitionId, nextSegmentRef.current);
  }, [beginDeviceFallback, cancelSpeech, clearAudioSource, exhibitionId, updateStatus]);

  useEffect(() => {
    if (!audioSupported || audioRef.current) return;
    const audio = new Audio();
    audio.preload = "auto";
    audioRef.current = audio;
  }, [audioSupported]);

  useEffect(() => {
    activeRef.current = active;
    segmentRef.current = segment;
    nextSegmentRef.current = nextSegment;
  }, [active, nextSegment, segment]);

  useEffect(() => {
    if (!speechSupported) return;
    const pickVoice = () => {
      const voices = window.speechSynthesis.getVoices();
      voiceRef.current =
        voices.find((voice) => /^zh(-|_)?CN/i.test(voice.lang)) ??
        voices.find((voice) => voice.lang.toLowerCase().startsWith("zh")) ??
        voices[0] ??
        null;
    };
    pickVoice();
    window.speechSynthesis.addEventListener("voiceschanged", pickVoice);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", pickVoice);
  }, [speechSupported]);

  // Stop the old segment before a late media event can affect the current one.
  useEffect(() => {
    const key = guideSegmentKey(segment);
    if (previousSegmentKeyRef.current === key) return;
    previousSegmentKeyRef.current = key;
    ++requestIdRef.current;
    fallbackRequestRef.current = null;
    cancelSpeech();
    clearAudioSource();
    updateStatus("paused");
    setMessage(null);

    if (selectedRef.current && active && segment.text.trim()) {
      beginNarration(segment);
    }
  }, [active, beginNarration, cancelSpeech, clearAudioSource, segment, updateStatus]);

  // Free walk and camera movement pause; arrival resumes the same stop.
  useEffect(() => {
    if (previousActiveRef.current === active) return;
    previousActiveRef.current = active;
    if (!active) {
      ++requestIdRef.current;
      fallbackRequestRef.current = null;
      audioRef.current?.pause();
      if (speechTimerRef.current !== null) {
        window.clearTimeout(speechTimerRef.current);
        speechTimerRef.current = null;
      }
      if ("speechSynthesis" in window) window.speechSynthesis.cancel();
      // State updates run after the media pause rather than synchronously in
      // the synchronization effect.
      window.setTimeout(() => {
        if (activeRef.current) return;
        updateFallbackSpeaking(false);
        updateStatus("paused");
        if (selectedRef.current) setMessage("导览移动或自由行走时语音已暂停");
      }, 0);
      return;
    }

    if (!selectedRef.current) return;
    const audio = audioRef.current;
    if (audio?.src && loadedSegmentRef.current === guideSegmentKey(segmentRef.current)) {
      const requestId = ++requestIdRef.current;
      fallbackRequestRef.current = null;
      audio.onplaying = () => {
        if (requestIdRef.current === requestId) updateStatus("playing");
      };
      audio.onended = () => {
        if (requestIdRef.current === requestId) updateStatus("paused");
      };
      audio.onerror = () => beginDeviceFallback(requestId, segmentRef.current);
      const playAttempt = audio.play();
      void playAttempt
        .then(() => {
          if (requestIdRef.current === requestId) {
            setMessage(null);
            updateStatus("playing");
          }
        })
        .catch((error: unknown) => {
          if (requestIdRef.current !== requestId) return;
          if (error instanceof DOMException && error.name === "NotAllowedError") {
            updateStatus("paused");
            setMessage("请点击“彦远专业讲述”继续播放");
          } else {
            beginDeviceFallback(requestId, segmentRef.current);
          }
        });
    } else {
      beginNarration(segmentRef.current);
    }
  }, [active, beginDeviceFallback, beginNarration, updateFallbackSpeaking, updateStatus]);

  useEffect(() => () => {
    ++requestIdRef.current;
    if (speechTimerRef.current !== null) window.clearTimeout(speechTimerRef.current);
    if ("speechSynthesis" in window) window.speechSynthesis.cancel();
    const audio = audioRef.current;
    if (audio) {
      audio.onplaying = null;
      audio.onended = null;
      audio.onerror = null;
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
    }
  }, []);

  const toggle = useCallback(() => {
    if (!supported) return;

    const isActivelyPlaying =
      statusRef.current === "preparing" ||
      statusRef.current === "playing" ||
      (statusRef.current === "fallback" && fallbackSpeakingRef.current);

    if (isActivelyPlaying) {
      selectedRef.current = false;
      ++requestIdRef.current;
      fallbackRequestRef.current = null;
      pauseAudio();
      setMessage(null);
      void logEvent("audio_guide_disabled", exhibitionId, { provider: "qwen_tts" });
      return;
    }

    selectedRef.current = true;
    setMessage(null);
    void logEvent("audio_guide_enabled", exhibitionId, { provider: "qwen_tts" });
    if (activeRef.current) beginNarration(segmentRef.current);
  }, [beginNarration, exhibitionId, pauseAudio, supported]);

  return {
    supported,
    status,
    message,
    fallbackSpeaking,
    toggle,
  };
}
