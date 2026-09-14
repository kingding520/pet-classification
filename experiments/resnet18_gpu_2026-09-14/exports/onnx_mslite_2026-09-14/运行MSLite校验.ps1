$liteRoot = 'D:\develop\pet-classification\MindSporePetClassification\mindspore-lite-2.9.0-win-x64'
$exportDir = Split-Path -Parent $PSCommandPath
$env:PATH = "$liteRoot\runtime\lib;$liteRoot\runtime\third_party\glog;$env:PATH"

& "$liteRoot\tools\benchmark\benchmark.exe" `
  --modelFile="$exportDir\resnet18_pet.ms" `
  --inDataFile="$exportDir\fixed_input.bin" `
  --benchmarkDataFile="$exportDir\expected_logits.bin" `
  --accuracyThreshold=5 `
  --device=CPU `
  --loopCount=1 `
  --warmUpLoopCount=0

exit $LASTEXITCODE
