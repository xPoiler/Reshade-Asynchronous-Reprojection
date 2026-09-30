# Installs (or removes) FrameWarp into any game that already has ReShade with full add-on support.
#   powershell -ExecutionPolicy Bypass -File tools\install.ps1 -Game "Expedition 33"        # Steam folder name (prefix ok)
#   powershell -ExecutionPolicy Bypass -File tools\install.ps1 -GameDir "D:\Games\Foo"       # game folder or its exe
#   powershell -ExecutionPolicy Bypass -File tools\install.ps1 -Game "Expedition 33" -Latewarp "C:\Downloads\nvngx_latewarp.dll"
#   powershell -ExecutionPolicy Bypass -File tools\install.ps1 -Game "Expedition 33" -Uninstall
# The files go next to ReShade (found by searching the folder). For Unreal Engine games the Streamline
# force-tagging cvar is added to the user Engine.ini; everything the installer changed is recorded in
# FrameWarp\install.json so -Uninstall reverts exactly that.
param(
    [string]$Game,
    [string]$GameDir,
    [string]$EngineIni,   # override the detected Unreal Engine.ini
    [switch]$NoCvar,      # do not touch Engine.ini
    [string]$Latewarp,    # path to nvngx_latewarp.dll (or a folder containing it)
    [switch]$Uninstall,
    [switch]$Force        # replace an existing FrameWarp without asking
)
$ErrorActionPreference = "Stop"
# Run from a release package (files next to this script) or from the source tree (build\Release).
$root = Split-Path -Parent $PSScriptRoot
$build = if (Test-Path (Join-Path $PSScriptRoot "FrameWarp.addon64")) { $PSScriptRoot } else { Join-Path $root "build\Release" }
$cvar = "r.Streamline.ForceTagging=1"
$reshadeNames = @("dxgi.dll", "d3d12.dll", "d3d11.dll", "dinput8.dll", "ReShade64.dll")
# Upscalers FrameWarp takes its data from: DLSS (Streamline, or called directly) and FSR 2 / 3 / 4 (their DLLs).
$upscalerNames = @("sl.interposer.dll", "nvngx_dlss.dll", "amd_fidelityfx_dx12.dll", "amd_fidelityfx_loader_dx12.dll",
                   "amd_fidelityfx_upscaler_dx12.dll", "ffx_fsr3upscaler_x64.dll", "ffx_fsr2_api_x64.dll")

function Get-SteamGameDirs {
    $steam = (Get-ItemProperty -ErrorAction SilentlyContinue "HKCU:\Software\Valve\Steam").SteamPath
    if (-not $steam) { $steam = "${env:ProgramFiles(x86)}\Steam" }
    $vdf = Join-Path $steam "steamapps\libraryfolders.vdf"
    $libraries = @($steam)
    if (Test-Path $vdf) {
        foreach ($m in [regex]::Matches((Get-Content -Raw $vdf), '"path"\s+"([^"]+)"')) { $libraries += $m.Groups[1].Value -replace '\\\\', '\' }
    }
    foreach ($lib in ($libraries | Select-Object -Unique)) {
        $common = Join-Path $lib "steamapps\common"
        if (Test-Path $common) { Get-ChildItem -Directory $common }
    }
}

# Version of a FrameWarp add-on, read from the "FrameWarp x.y.z" string it carries (every release does).
function Get-FrameWarpVersion([string]$path) {
    if (-not $path -or -not (Test-Path -PathType Leaf $path)) { return $null }
    $text = [Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes($path))
    $m = [regex]::Match($text, 'FrameWarp (\d+\.\d+\.\d+)')
    if ($m.Success) { return $m.Groups[1].Value } else { return "unknown" }
}
$packageVersion = Get-FrameWarpVersion (Join-Path $build "FrameWarp.addon64")

function Test-ReShade([string]$path) {
    $info = (Get-Item $path).VersionInfo
    return ($info.ProductName -match "ReShade") -or ($info.FileDescription -match "ReShade")
}

# ReShade for Vulkan games is a system-wide Vulkan layer, not a DLL in the game folder: the game folder
# only gets ReShade.ini next to the game's executable, and ReShade loads add-ons from there.
function Test-VulkanReShadeLayer {
    foreach ($key in @("HKLM:\SOFTWARE\Khronos\Vulkan\ImplicitLayers", "HKCU:\SOFTWARE\Khronos\Vulkan\ImplicitLayers")) {
        $layers = Get-ItemProperty -ErrorAction SilentlyContinue $key
        if ($layers -and @($layers.PSObject.Properties | Where-Object { $_.Name -match "ReShade[^\\]*\.json$" }).Count -gt 0) { return $true }
    }
    return $false
}
$vulkanLayer = Test-VulkanReShadeLayer
# Folders with a ReShade.ini next to an executable (the Vulkan layer's per-game settings).
function Get-VulkanReShadeDirs([object[]]$files) {
    if (-not $vulkanLayer) { return @() }
    return @($files | Where-Object { $_.Name -eq "ReShade.ini" } | Select-Object -ExpandProperty DirectoryName -Unique |
             Where-Object { Get-ChildItem -File -Filter "*.exe" $_ -ErrorAction SilentlyContinue })
}

# Resolve the game folder.
if ($GameDir) {
    if (Test-Path -PathType Leaf $GameDir) { $GameDir = Split-Path -Parent $GameDir }
    if (-not (Test-Path $GameDir)) { throw "Folder not found: $GameDir" }
} elseif ($Game) {
    $all = @(Get-SteamGameDirs | Sort-Object FullName -Unique)
    $found = @($all | Where-Object { $_.Name -eq $Game })
    if (-not $found) { $found = @($all | Where-Object { $_.Name -like "$Game*" }) }
    if ($found.Count -eq 0) { throw "No Steam game folder matches '$Game'. Installed: $(($all | ForEach-Object Name) -join ', ')" }
    if ($found.Count -gt 1) { throw "'$Game' matches several games: $(($found | ForEach-Object Name) -join ', ')" }
    $GameDir = $found[0].FullName
} else {
    # No game given (double-clicked install.bat / uninstall.bat): list the Steam games that can be chosen.
    Write-Host "Scanning Steam libraries..."
    $choices = @()
    foreach ($dir in @(Get-SteamGameDirs | Sort-Object FullName -Unique)) {
        $found = @(Get-ChildItem -Recurse -Depth 8 -File -ErrorAction SilentlyContinue $dir.FullName -Include ($reshadeNames + $upscalerNames + "FrameWarp.addon64" + "ReShade.ini"))
        $hasReShade = [bool]($found | Where-Object { $reshadeNames -contains $_.Name } | Where-Object { Test-ReShade $_.FullName }) -or
                      ((Get-VulkanReShadeDirs $found).Count -gt 0)
        # The installed add-on (not a staged copy): the one closest to the game folder.
        $installedAddon = $found | Where-Object { $_.Name -eq "FrameWarp.addon64" } | Sort-Object { $_.FullName.Split('\').Count } | Select-Object -First 1
        $hasFrameWarp = [bool]$installedAddon
        $installedVersion = if ($installedAddon) { Get-FrameWarpVersion $installedAddon.FullName } else { $null }
        # DLSS (through Streamline or called directly) or FSR (its DLLs): FrameWarp works with any of them.
        $hasUpscaler = [bool]($found | Where-Object { $upscalerNames -contains $_.Name })
        if (($Uninstall -and $hasFrameWarp) -or (-not $Uninstall -and $hasReShade)) {
            $choices += [pscustomobject]@{ Name = $dir.Name; Path = $dir.FullName; FrameWarp = $hasFrameWarp; Version = $installedVersion; Upscaler = $hasUpscaler }
        }
    }
    Write-Host ""
    if ($choices.Count -eq 0) {
        Write-Host $(if ($Uninstall) { "No Steam game with FrameWarp installed was found." } else { "No Steam game with ReShade was found. Install ReShade (with full add-on support) for the game first." })
    } else {
        Write-Host $(if ($Uninstall) { "Games with FrameWarp installed:" } else { "Games with ReShade:" })
        for ($i = 0; $i -lt $choices.Count; ++$i) {
            $c = $choices[$i]
            $notes = @()
            if (-not $Uninstall) {
                if ($c.FrameWarp) { $notes += "FrameWarp $($c.Version) installed" }
                if (-not $c.Upscaler) { $notes += "no DLSS or FSR found: will not work" }
            }
            if ($Uninstall -and $c.Version) { $notes += "FrameWarp $($c.Version)" }
            Write-Host ("  {0,2}) {1}{2}" -f ($i + 1), $c.Name, $(if ($notes) { "   [" + ($notes -join ", ") + "]" } else { "" }))
        }
    }
    Write-Host "   P) Enter a game folder by hand (non-Steam games)"
    Write-Host "   Q) Quit"
    while (-not $GameDir) {
        $answer = (Read-Host "Choose").Trim()
        if ($answer -match '^[Qq]$') { Write-Host "Nothing changed."; return }
        if ($answer -match '^[Pp]$') {
            $typed = (Read-Host "Game folder (or the game's .exe)").Trim().Trim('"')
            if (Test-Path -PathType Leaf $typed) { $typed = Split-Path -Parent $typed }
            if ($typed -and (Test-Path -PathType Container $typed)) { $GameDir = $typed } else { Write-Host "Folder not found." }
            continue
        }
        $n = 0
        if ([int]::TryParse($answer, [ref]$n) -and $n -ge 1 -and $n -le $choices.Count) { $GameDir = $choices[$n - 1].Path }
        else { Write-Host "Type a number from the list, P or Q." }
    }
    Write-Host ""
}

# ReShade's folder is where the add-on must go (it loads add-ons from its own directory).
$reshade = @(Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue $GameDir -Include $reshadeNames | Where-Object { Test-ReShade $_.FullName })
$reshadeKind = if ($reshade.Count -gt 0) { $reshade[0].Name } else { $null }
if ($reshade.Count -eq 0 -and -not $Uninstall) {
    # Vulkan games: ReShade's layer, with the game's ReShade.ini next to its executable.
    $vkDirs = @(Get-VulkanReShadeDirs @(Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue $GameDir -Filter "ReShade.ini"))
    if ($vkDirs.Count -gt 1) { throw "ReShade.ini found next to several executables: $($vkDirs -join ', '). Pass the right folder with -GameDir." }
    if ($vkDirs.Count -eq 1) {
        $reshade = @(Get-Item (Join-Path $vkDirs[0] "ReShade.ini"))
        $reshadeKind = "Vulkan layer"
    }
}
if ($reshade.Count -eq 0) {
    if ($Uninstall) { $installed = @(Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue $GameDir -Filter "FrameWarp.addon64") }
    if (-not $installed) {
        throw "ReShade not found under $GameDir (looked for $($reshadeNames -join ', ') made by ReShade, or, for Vulkan games, ReShade's Vulkan layer with a ReShade.ini next to the game's executable)."
    }
    $binDir = $installed[0].DirectoryName
} else {
    $dirs = @($reshade | Select-Object -ExpandProperty DirectoryName -Unique)
    if ($dirs.Count -gt 1) {
        # Mod managers keep backup copies (e.g. a _storage_ folder). The live ReShade sits next to the
        # game's executable; among those, take the one closest to the game folder.
        $withExe = @($dirs | Where-Object { Get-ChildItem -File -Filter "*.exe" $_ -ErrorAction SilentlyContinue })
        if ($withExe.Count -gt 0) { $dirs = $withExe }
        $depth = { param($d) ($d.TrimEnd('\') -split '\\').Count }
        $minDepth = ($dirs | ForEach-Object { & $depth $_ } | Measure-Object -Minimum).Minimum
        $dirs = @($dirs | Where-Object { (& $depth $_) -eq $minDepth })
        if ($dirs.Count -gt 1) {
            throw "ReShade found in several folders: $(($reshade | ForEach-Object FullName) -join ', '). Pass the right one with -GameDir."
        }
    }
    $binDir = $dirs[0]
    $reshade = @($reshade | Where-Object { $_.DirectoryName -eq $binDir })
}
$target = Join-Path $binDir "FrameWarp"
$record = Join-Path $target "install.json"

# An existing FrameWarp is only replaced after asking (or with -Force).
if (-not $Uninstall -and -not $Force) {
    $existing = Get-FrameWarpVersion (Join-Path $binDir "FrameWarp.addon64")
    if ($existing) {
        $question = "FrameWarp $existing is installed. Update it to $packageVersion?"
        try {
            if ($existing -ne "unknown" -and $packageVersion -and $packageVersion -ne "unknown") {
                if ([version]$existing -eq [version]$packageVersion) { $question = "FrameWarp $existing is already installed. Reinstall it?" }
                elseif ([version]$existing -gt [version]$packageVersion) { $question = "FrameWarp $existing is installed, which is NEWER than this package ($packageVersion). Replace it with the older version?" }
            }
        } catch {}
        $answer = (Read-Host "$question (Y/N)").Trim()
        if ($answer -notmatch '^[Yy]') { Write-Host "Nothing changed."; return }
    }
}

# The game must not be running (its exe lives in or below the game folder). A presenter left over from
# a crashed game session would lock its exe and break the update: stop it (it only serves the game).
$inGame = @(Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.Path -and $_.Path.StartsWith($GameDir, [StringComparison]::OrdinalIgnoreCase) })
$running = @($inGame | Where-Object { $_.ProcessName -ne "FrameWarpPresenter" })
if ($running) { throw "Close the game first ($(($running | ForEach-Object ProcessName | Select-Object -Unique) -join ', '))." }
$leftover = @($inGame | Where-Object { $_.ProcessName -eq "FrameWarpPresenter" })
if ($leftover) {
    $leftover | Stop-Process -Force -ErrorAction SilentlyContinue
    $leftover | ForEach-Object { $_.WaitForExit(5000) | Out-Null }
    Write-Host "Stopped a FrameWarp presenter left running from an earlier game session."
}

# Unreal Engine layout: <Game>\<Project>\Binaries\Win64\<exe>. The user config lives in
# %LOCALAPPDATA%\<Project>\Saved\Config\Windows (UE5), WindowsNoEditor (UE4) or WinGDK (Game Pass).
function Find-EngineIni {
    if ((Split-Path -Leaf $binDir) -notmatch '^Win64$|^WinGDK$') { return $null }
    $project = Split-Path -Leaf (Split-Path -Parent (Split-Path -Parent $binDir))
    # Usually <Project>\Saved, but some games add a store level (Returnal: Returnal\Steam\Saved).
    foreach ($base in @((Join-Path $env:LOCALAPPDATA $project)) + @(Get-ChildItem -Directory -ErrorAction SilentlyContinue (Join-Path $env:LOCALAPPDATA $project) | ForEach-Object FullName)) {
        foreach ($platform in @("Windows", "WindowsNoEditor", "WinGDK")) {
            $dir = Join-Path $base "Saved\Config\$platform"
            if (Test-Path $dir) { return Join-Path $dir "Engine.ini" }
        }
    }
    return $null
}

function Set-Cvar([string]$ini, [bool]$enable) {
    if (-not (Test-Path $ini)) { if ($enable) { New-Item -ItemType File -Force $ini | Out-Null } else { return } }
    $backup = "$ini.framewarp-backup"
    if ($enable -and -not (Test-Path $backup)) { Copy-Item $ini $backup }
    $lines = [System.Collections.Generic.List[string]](@(Get-Content $ini))
    $lines.RemoveAll({ param($l) $l.Trim() -eq $cvar }) | Out-Null
    if ($enable) {
        $i = $lines.FindIndex({ param($l) $l.Trim() -eq "[ConsoleVariables]" })
        if ($i -lt 0) { $lines.Add(""); $lines.Add("[ConsoleVariables]"); $lines.Add($cvar) } else { $lines.Insert($i + 1, $cvar) }
    }
    # Performance mods often mark Engine.ini read-only so the game cannot rewrite it; keep that.
    $item = Get-Item $ini
    $readOnly = $item.IsReadOnly
    if ($readOnly) { $item.IsReadOnly = $false }
    try { Set-Content -Path $ini -Value $lines -Encoding UTF8 }
    finally { if ($readOnly) { (Get-Item $ini).IsReadOnly = $true } }
}

if ($Uninstall) {
    $info = if (Test-Path $record) { Get-Content -Raw $record | ConvertFrom-Json } else { $null }
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $binDir "FrameWarp.addon64")
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $binDir "reshade-shaders\Shaders\XPAR.fx")
    # Some games copy their DLLs into a staging folder at launch (RE9: _storage_); remove those copies too.
    Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue $GameDir -Filter "FrameWarp.addon64" | Remove-Item -Force -ErrorAction SilentlyContinue
    $old = Join-Path $target "disabled\ReprojectionDiagnostics.addon64"
    if (Test-Path $old) { Move-Item -Force $old $binDir }
    if ($info -and $info.engine_ini) { Set-Cvar $info.engine_ini $false; Write-Host "Engine.ini: removed $cvar ($($info.engine_ini))" }
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $target
    Write-Host "FrameWarp removed from $binDir"
    return
}

foreach ($f in @("FrameWarp.addon64", "FrameWarp\FrameWarpPresenter.exe")) {
    if (-not (Test-Path (Join-Path $build $f))) { throw "Missing $f in $build - build Release first." }
}
# NVIDIA Latewarp is optional (FrameWarp's own warp engine is used without it). nvngx_latewarp.dll is
# NVIDIA's and not redistributable: take it from -Latewarp, the package/build folder,
# third_party\latewarp in the source tree, or an existing install.
$latewarpCandidates = @()
if ($Latewarp) { $latewarpCandidates += $(if (Test-Path -PathType Container $Latewarp) { Join-Path $Latewarp "nvngx_latewarp.dll" } else { $Latewarp }) }
$latewarpCandidates += @((Join-Path $build "FrameWarp\nvngx_latewarp.dll"), (Join-Path $build "nvngx_latewarp.dll"),
                         (Join-Path $root "third_party\latewarp\nvngx_latewarp.dll"),
                         (Join-Path $target "nvngx_latewarp.dll"))
$latewarpDll = $latewarpCandidates | Where-Object { Test-Path -PathType Leaf $_ } | Select-Object -First 1
if (-not (Test-Path (Join-Path $env:SystemRoot "System32\msvcp140.dll"))) {
    Write-Warning "Microsoft Visual C++ 2015-2022 Redistributable (x64) not found: the presenter needs it (https://aka.ms/vs/17/release/vc_redist.x64.exe)."
}
$previous = if (Test-Path $record) { Get-Content -Raw $record | ConvertFrom-Json } else { $null }
# Some game folders (for example ones installed by the Rockstar Games Launcher) only let administrators
# write: say so plainly instead of failing halfway.
try {
    New-Item -ItemType Directory -Force $target -ErrorAction Stop | Out-Null
    $probe = Join-Path $target ".write-test"
    Set-Content -Path $probe -Value "" -ErrorAction Stop
    Remove-Item -Force $probe
} catch [System.UnauthorizedAccessException] {
    throw "Windows does not allow writing to $binDir without administrator rights. Right-click install.bat and choose 'Run as administrator'."
}
Copy-Item -Force (Join-Path $build "FrameWarp.addon64") $binDir
# Some games copy their DLLs into a staging folder at launch and load them from there (RE9: _storage_);
# replace those copies too, so an update never depends on the game refreshing them.
Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue $GameDir -Filter "FrameWarp.addon64" |
    Where-Object { $_.DirectoryName -ne $binDir } |
    ForEach-Object { Copy-Item -Force (Join-Path $build "FrameWarp.addon64") $_.FullName; Write-Host "Updated staged copy: $($_.FullName)" }
Copy-Item -Force (Join-Path $build "FrameWarp\FrameWarpPresenter.exe") $target
# XPAR.fx (depth from ReShade, for games without DLSS or FSR) goes where ReShade keeps
# its effects; without that folder ReShade has no ReShade.fxh for it either, so it is left out.
$feed = @((Join-Path $build "XPAR.fx"), (Join-Path $root "shaders\XPAR.fx")) | Where-Object { Test-Path -PathType Leaf $_ } | Select-Object -First 1
$shaderDir = Join-Path $binDir "reshade-shaders\Shaders"
$feedInstalled = $false
if ($feed -and (Test-Path -PathType Container $shaderDir)) { Copy-Item -Force $feed $shaderDir; $feedInstalled = $true }
$latewarpTarget = Join-Path $target "nvngx_latewarp.dll"
if ($latewarpDll -and (Resolve-Path $latewarpDll).Path -ne $latewarpTarget) { Copy-Item -Force $latewarpDll $latewarpTarget }
# The previous prototype's add-on captures every frame through an effect; park it (reversible).
$old = Join-Path $binDir "ReprojectionDiagnostics.addon64"
if (Test-Path $old) {
    New-Item -ItemType Directory -Force (Join-Path $target "disabled") | Out-Null
    Move-Item -Force $old (Join-Path $target "disabled")
}
Write-Host "Installed to $binDir (ReShade: $reshadeKind)"
if ($feedInstalled) { Write-Host "XPAR.fx: installed in $shaderDir (games without DLSS or FSR: set up ReShade's depth buffer)" }
if ($latewarpDll) { Write-Host "NVIDIA Latewarp: installed (optional warp engine, NVIDIA GPUs)" }
else { Write-Host "NVIDIA Latewarp: not included (optional) - FrameWarp's own warp engine is used" }

# FrameWarp takes its data from DLSS (through Streamline: camera, depth, motion and HUD tags; or called
# directly: depth and motion vectors, the presenter works out the camera) or from FSR (its DLLs).
$streamline = Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue $GameDir -Filter "sl.interposer.dll" | Select-Object -First 1
$dlss = Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue $GameDir -Filter "nvngx_dlss.dll" | Select-Object -First 1
$fsr = @(Get-ChildItem -Recurse -File -ErrorAction SilentlyContinue $GameDir -Include ($upscalerNames | Where-Object { $_ -match "fidelityfx|ffx_" })) | Select-Object -First 1
if ($streamline) { Write-Host "Streamline: $($streamline.FullName)" }
elseif ($dlss) { Write-Host "DLSS without Streamline: $($dlss.FullName) (the camera is worked out from DLSS's motion vectors)" }
if ($fsr) { Write-Host "FSR: $($fsr.FullName)" }
if (-not $streamline -and -not $dlss -and -not $fsr) {
    Write-Warning "No DLSS or FSR found: FrameWarp needs a game with DLSS, or with FSR 2 / 3 / 4 shipped as a DLL."
}

$ini = $null
if (-not $NoCvar) {
    $ini = if ($EngineIni) { $EngineIni } elseif ($previous -and $previous.engine_ini) { $previous.engine_ini } else { Find-EngineIni }
    if ($ini) {
        Set-Cvar $ini $true
        Write-Host "Engine.ini: $cvar under [ConsoleVariables] ($ini, backup: $ini.framewarp-backup)"
    } elseif ((Split-Path -Leaf $binDir) -match '^Win64$|^WinGDK$') {
        Write-Warning "Looks like Unreal Engine but no user config folder was found (run the game once, or pass -EngineIni)."
    }
    if ($streamline -and ($ini -or (Split-Path -Leaf $binDir) -match '^Win64$|^WinGDK$')) {
        Write-Host "Unreal Engine + Streamline: add launch options  -slforcetagging -slviewextension"
    }
}
@{ game_dir = $GameDir; bin_dir = $binDir; engine_ini = $ini; installed = (Get-Date -Format s) } | ConvertTo-Json | Set-Content -Encoding UTF8 $record
