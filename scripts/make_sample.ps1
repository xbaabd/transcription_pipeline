# Generates a short spoken WAV fixture using the Windows built-in TTS engine,
# so the repo has a test input without shipping third-party audio.
Add-Type -AssemblyName System.Speech
$root = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force -Path "$root\samples" | Out-Null
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile("$root\samples\sample_short.wav")
$synth.Speak("Hello, this is a short test recording for the transcription pipeline. It contains two sentences spoken clearly, followed by a final closing remark.")
$synth.Dispose()
Write-Host "sample written to samples\sample_short.wav"
