$ErrorActionPreference = "Stop"

$tesseract = "C:\Program Files\Tesseract-OCR\tesseract.exe"
if (-not (Test-Path -LiteralPath $tesseract)) {
    winget install --id UB-Mannheim.TesseractOCR --exact --silent `
        --accept-package-agreements --accept-source-agreements --disable-interactivity
}
if (-not (Test-Path -LiteralPath $tesseract)) {
    throw "Tesseract no quedó disponible en $tesseract"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$tessdata = Join-Path $repoRoot ".cache\tessdata"
New-Item -ItemType Directory -Force -Path $tessdata | Out-Null
$revision = "87416418657359cb625c412a48b6e1d6d41c29bd"
foreach ($language in @("spa", "eng", "por", "osd")) {
    $url = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/$revision/$language.traineddata"
    Invoke-WebRequest -Uri $url -OutFile (Join-Path $tessdata "$language.traineddata")
}

& $tesseract --version
& $tesseract --list-langs --tessdata-dir $tessdata
Write-Output "OCR listo. tessdata=$tessdata"
