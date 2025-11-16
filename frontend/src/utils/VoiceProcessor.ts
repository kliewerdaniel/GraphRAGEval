/**
 * Voice Processor for enhancing speech synthesis
 * Optimized for speechSynthesis API without Web Audio API complexity
 */
export class VoiceProcessor {
  // Voice enhancement settings
  private settings = {
    volumeNormalization: true,
    rateSmoothing: true,
    pitchModulation: true,
  }

  /**
   * Enhance speech synthesis utterance parameters for more natural sound
   * Note: speechSynthesis API doesn't support Web Audio API filtering
   */
  async enhanceSpeech(utterance: SpeechSynthesisUtterance, text?: string): Promise<void> {
    if (!this.settings.volumeNormalization &&
        !this.settings.rateSmoothing &&
        !this.settings.pitchModulation) {
      return; // Skip enhancement if disabled
    }

    // Store original parameters
    const originalRate = utterance.rate;
    const originalPitch = utterance.pitch;
    const originalVolume = utterance.volume;

    // Apply subtle volume normalization for consistent loudness
    if (this.settings.volumeNormalization) {
      utterance.volume = Math.min(originalVolume * 1.05, 0.9); // Very conservative boost
    }

    // Apply rate smoothing based on content for clearer pronunciation
    if (this.settings.rateSmoothing) {
      // Slightly reduce rate for clearer speech
      utterance.rate = Math.max(0.85, Math.min(1.0, originalRate * 0.95));

      // Adjust rate based on text complexity
      if (text) {
        const wordCount = text.trim().split(/\s+/).length;
        const longWords = (text.match(/\b\w{8,}\b/g) || []).length;
        const sentences = (text.match(/[.!?]+/g) || []).length || 1;

        // Complex content gets slower rate for clarity
        const complexityScore = (wordCount / 20 + longWords * 0.5) / sentences;
        const rateAdjustment = Math.min(complexityScore * 0.03, 0.05);

        utterance.rate = Math.max(0.75, utterance.rate * (1 - rateAdjustment));
      }
    }

    // Add subtle pitch modulation for more natural expression
    if (this.settings.pitchModulation) {
      // Use time-based variation for consistency across utterances
      const timeSeed = Math.floor(Date.now() / 5000) * 0.01; // Slow variation
      const pitchVariation = (Math.sin(timeSeed) * 0.5 + 0.5) * 0.05; // Very subtle
      utterance.pitch = Math.max(0.9, Math.min(1.1, originalPitch + pitchVariation));
    }
  }

  /**
   * Update voice enhancement settings
   */
  updateSettings(newSettings: Partial<typeof this.settings>) {
    this.settings = { ...this.settings, ...newSettings };
  }

  /**
   * Get current enhancement settings
   */
  getSettings() {
    return { ...this.settings };
  }

  /**
   * Check if voice processor is ready (always true for this simplified version)
   */
  isReady(): boolean {
    return true;
  }
}

// Singleton instance for global use
export const voiceProcessor = new VoiceProcessor();
