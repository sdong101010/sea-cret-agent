// Speech sidecar for sea-cret-agent.
//
// Reads raw 16-bit signed little-endian PCM audio @ 16kHz mono on stdin,
// streams it through Apple's SpeechAnalyzer (macOS 26+), and writes
// finalized transcript segments as JSONL on stdout, one per line:
//
//   {"text": "hello there", "start": 0.42, "end": 1.85, "is_final": true}
//
// Build:
//   swiftc -O bin/speech_sidecar.swift -o bin/speech_sidecar
//
// Requires macOS 26+ (Tahoe) for SpeechAnalyzer.

import Foundation
import Speech
import AVFoundation
import CoreMedia

let SAMPLE_RATE: Double = 16000

// stderr logging helper.
func slog(_ msg: String) {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
}

// Emit a JSONL line on stdout, then flush.
func emit(_ obj: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: obj, options: []) else {
        return
    }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

// Wrap Int16 PCM bytes in an AVAudioPCMBuffer of Int16 @ 16kHz mono.
// SpeechAnalyzer requires 16-bit signed integer samples.
func makeBuffer(from data: Data, format: AVAudioFormat) -> AVAudioPCMBuffer? {
    let sampleCount = data.count / 2
    guard sampleCount > 0 else { return nil }
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                        frameCapacity: AVAudioFrameCount(sampleCount)) else {
        return nil
    }
    buffer.frameLength = AVAudioFrameCount(sampleCount)
    let dst = buffer.int16ChannelData![0]
    data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
        let src = raw.bindMemory(to: Int16.self)
        for i in 0..<sampleCount {
            dst[i] = src[i]
        }
    }
    return buffer
}

@available(macOS 26.0, *)
func runSpeechAnalyzer() async throws {
    // Note: do NOT call SFSpeechRecognizer.requestAuthorization here — that
    // triggers a TCC check requiring NSSpeechRecognitionUsageDescription in
    // Info.plist (which a bare CLI binary doesn't have). The new on-device
    // SpeechAnalyzer path on macOS 26+ does not require that authorization.

    let locale = Locale(identifier: "en-US")
    let transcriber = SpeechTranscriber(
        locale: locale,
        transcriptionOptions: [],
        reportingOptions: [.volatileResults],
        attributeOptions: []
    )

    // Ensure the on-device model is downloaded.
    if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
        slog("[sidecar] downloading speech assets (one-time)...")
        try await request.downloadAndInstall()
        slog("[sidecar] speech assets ready")
    }

    let format = AVAudioFormat(commonFormat: .pcmFormatInt16,
                               sampleRate: SAMPLE_RATE,
                               channels: 1,
                               interleaved: false)!

    let (inputSequence, inputContinuation) = AsyncStream.makeStream(of: AnalyzerInput.self)

    let analyzer = SpeechAnalyzer(modules: [transcriber])

    try await analyzer.start(inputSequence: inputSequence)
    slog("[sidecar] SpeechAnalyzer started")

    // Result consumer — emit segments as they arrive.
    let resultsTask = Task {
        do {
            for try await result in transcriber.results {
                let text = String(result.text.characters).trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty else { continue }
                let start = result.range.start.seconds
                let end = (result.range.start + result.range.duration).seconds
                emit([
                    "text": text,
                    "start": start.isFinite ? start : 0.0,
                    "end": end.isFinite ? end : 0.0,
                    "is_final": result.isFinal,
                ])
            }
        } catch {
            slog("[sidecar] result stream error: \(error)")
        }
    }

    // Stdin reader. Read in ~100ms chunks: 16000Hz * 0.1s * 2 bytes = 3200 bytes.
    let stdin = FileHandle.standardInput
    let CHUNK = 3200
    while true {
        let data = stdin.availableData
        if data.isEmpty {
            break  // EOF
        }
        var buf = data
        while buf.count < CHUNK {
            let more = stdin.availableData
            if more.isEmpty { break }
            buf.append(more)
        }
        if let pcm = makeBuffer(from: buf, format: format) {
            inputContinuation.yield(AnalyzerInput(buffer: pcm))
        }
    }

    slog("[sidecar] stdin EOF, flushing...")
    inputContinuation.finish()
    try await analyzer.finalizeAndFinishThroughEndOfInput()
    _ = await resultsTask.value
    slog("[sidecar] done")
}

// Entry point
if #available(macOS 26.0, *) {
    do {
        try await runSpeechAnalyzer()
    } catch {
        slog("[sidecar] fatal: \(error)")
        emit(["error": "fatal", "message": "\(error)"])
        exit(1)
    }
} else {
    slog("[sidecar] SpeechAnalyzer requires macOS 26+. Current OS does not support this transcriber.")
    emit(["error": "unsupported_os"])
    exit(3)
}
