# Builds the release package: dist\FrameWarp-<version>\ and dist\FrameWarp-<version>.zip.
#   powershell -ExecutionPolicy Bypass -File tools\package.ps1
# nvngx_latewarp.dll (NVIDIA Reflex 2 Frame Warp) is not redistributable and is left out.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$version = [regex]::Match((Get-Content -Raw (Join-Path $root "CMakeLists.txt")), 'project\(FrameWarp VERSION ([0-9.]+)').Groups[1].Value
if (-not $version) { throw "Version not found in CMakeLists.txt" }

cmake --build (Join-Path $root "build") --config Release
if ($LASTEXITCODE) { throw "Build failed" }
ctest --test-dir (Join-Path $root "build") -C Release --output-on-failure
if ($LASTEXITCODE) { throw "Tests failed" }

$build = Join-Path $root "build\Release"
$name = "FrameWarp-$version"
$dist = Join-Path $root "dist"
$out = Join-Path $dist $name
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $out, "$out.zip"
New-Item -ItemType Directory -Force (Join-Path $out "FrameWarp") | Out-Null
Copy-Item (Join-Path $build "FrameWarp.addon64") $out
Copy-Item (Join-Path $build "FrameWarp\FrameWarpPresenter.exe") (Join-Path $out "FrameWarp")
Copy-Item (Join-Path $root "shaders\XPAR.fx") $out
Copy-Item (Join-Path $root "tools\install.ps1") $out
Copy-Item (Join-Path $root "tools\release\install.bat") $out
Copy-Item (Join-Path $root "tools\release\uninstall.bat") $out
Copy-Item (Join-Path $root "README.md") $out
Copy-Item (Join-Path $root "CHANGELOG.md") $out
Copy-Item (Join-Path $root "THIRD_PARTY_NOTICES.md") $out
if (Test-Path (Join-Path $root "LICENSE")) { Copy-Item (Join-Path $root "LICENSE") $out }

# Windows' bsdtar writes standard zips (Compress-Archive in PowerShell 5.1 uses backslash separators).
& "$env:SystemRoot\System32\tar.exe" -a -c -f "$out.zip" -C $dist $name
if ($LASTEXITCODE) { throw "zip failed" }
$hash = (Get-FileHash -Algorithm SHA256 "$out.zip").Hash.ToLower()
"$hash  $name.zip" | Set-Content -Encoding ASCII (Join-Path $dist "$name.zip.sha256")
Get-ChildItem -Recurse $out | ForEach-Object { $_.FullName.Substring($out.Length + 1) }
Write-Host "Package: $out.zip"
Write-Host "SHA256:  $hash"
