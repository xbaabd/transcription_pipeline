# Generates a ~3 minute spoken WAV with numbered sentences, used to verify
# that chunked transcription keeps timestamps globally correct.
Add-Type -AssemblyName System.Speech
$root = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force -Path "$root\samples" | Out-Null
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile("$root\samples\sample_long.wav")
foreach ($i in 1..40) {
    $synth.Speak("This is sentence number $i of the long recording test. The pipeline should keep every timestamp accurate across chunk boundaries.")
}
$synth.Dispose()
Write-Host "long sample written to samples\sample_long.wav"
