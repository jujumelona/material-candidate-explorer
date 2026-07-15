[CmdletBinding()]
param(
    [ValidateSet("core", "fusion", "molecule-modern", "molecule-generation", "materials-open", "biology-open", "electronic-open", "uma", "all-open", "all")]
    [string]$Profile = "core",

    [ValidateSet("auto", "cpu", "cuda", "mps")]
    [string]$Accelerator = "auto",

    [ValidateSet("auto", "native", "wsl")]
    [string]$Backend = "auto",

    [switch]$IncludeWeights,
    [string[]]$AcceptLicense = @(),
    [switch]$DryRun,
    [switch]$RequireAll,
    [string]$Manifest,
    [switch]$AllowCustomManifest,
    [string]$InstallRoot,
    [switch]$AllowExternalRoot
)

$ErrorActionPreference = "Stop"
if (Test-Path -LiteralPath "variable:PSNativeCommandUseErrorActionPreference") {
    $PSNativeCommandUseErrorActionPreference = $false
}
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $root "scripts\bootstrap.py"

$linuxProfiles = @("materials-open", "biology-open", "electronic-open", "uma", "all-open", "all")
$wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
$wslUsable = $false
if ($null -ne $wsl) {
    & $wsl.Source --exec python3 -c "import ensurepip, venv" *> $null
    $wslUsable = ($LASTEXITCODE -eq 0)
}
if ($Backend -eq "auto") {
    if (($Profile -in $linuxProfiles) -and $wslUsable) {
        $Backend = "wsl"
    } else {
        $Backend = "native"
    }
}

$arguments = @("install", "--profile", $Profile, "--accelerator", $Accelerator)
if ($IncludeWeights) { $arguments += "--include-weights" }
if ($DryRun) { $arguments += "--dry-run" }
if ($RequireAll) { $arguments += "--require-all" }
if ($AllowCustomManifest) { $arguments += "--allow-custom-manifest" }
if ($AllowExternalRoot) { $arguments += "--allow-external-root" }
foreach ($license in $AcceptLicense) {
    $arguments += @("--accept-license", $license)
}

if ($Backend -eq "wsl") {
    if (-not $wslUsable) {
        throw "-Backend wsl requires a configured Linux distribution with python3, venv, and ensurepip."
    }
    $wslRoot = (& $wsl.Source --exec wslpath -a $root).Trim()
    if ([string]::IsNullOrWhiteSpace($wslRoot)) {
        throw "Could not translate the workspace path into WSL."
    }
    $wslScript = "$wslRoot/scripts/bootstrap.py"
    if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
        $backendRoot = "$wslRoot/.discovery/wsl"
    } elseif ($InstallRoot.StartsWith("/")) {
        $backendRoot = $InstallRoot
    } else {
        $windowsInstallRoot = if ([System.IO.Path]::IsPathRooted($InstallRoot)) {
            [System.IO.Path]::GetFullPath($InstallRoot)
        } else {
            [System.IO.Path]::GetFullPath((Join-Path $root $InstallRoot))
        }
        $backendRoot = (& $wsl.Source --exec wslpath -a $windowsInstallRoot).Trim()
        if ([string]::IsNullOrWhiteSpace($backendRoot)) {
            throw "Could not translate -InstallRoot into WSL."
        }
    }
    if (-not [string]::IsNullOrWhiteSpace($Manifest)) {
        if ($Manifest.StartsWith("/")) {
            $wslManifest = $Manifest
        } else {
            $windowsManifest = if ([System.IO.Path]::IsPathRooted($Manifest)) {
                [System.IO.Path]::GetFullPath($Manifest)
            } else {
                [System.IO.Path]::GetFullPath((Join-Path $root $Manifest))
            }
            $wslManifest = (& $wsl.Source --exec wslpath -a $windowsManifest).Trim()
            if ([string]::IsNullOrWhiteSpace($wslManifest)) {
                throw "Could not translate -Manifest into WSL."
            }
        }
        $arguments += @("--manifest", $wslManifest)
    }
    $arguments += @("--root", $backendRoot)
    # WSL does not reliably inherit arbitrary Windows variables unless they are
    # named in WSLENV. Pass only the reviewed download credential/acceptance
    # variables for this process; never put their values in argv or state files.
    $savedWslenv = $env:WSLENV
    try {
        $bridgeNames = @()
        foreach ($name in @("HF_TOKEN", "ACCEPT_ESM_LICENSE", "ACCEPT_UMA_LICENSE")) {
            if (-not [string]::IsNullOrWhiteSpace((Get-Item "Env:$name" -ErrorAction SilentlyContinue).Value)) {
                $bridgeNames += $name
            }
        }
        $combinedWslenv = @($savedWslenv, $bridgeNames) | Where-Object {
            -not [string]::IsNullOrWhiteSpace($_)
        }
        if ($combinedWslenv.Count -gt 0) { $env:WSLENV = $combinedWslenv -join ':' }
        & $wsl.Source --exec python3 $wslScript @arguments
        $wslExitCode = $LASTEXITCODE
    } finally {
        if ($null -eq $savedWslenv) { Remove-Item Env:WSLENV -ErrorAction SilentlyContinue }
        else { $env:WSLENV = $savedWslenv }
    }
    exit $wslExitCode
}

$activePython = if (-not [string]::IsNullOrWhiteSpace($env:VIRTUAL_ENV)) {
    Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"
} else {
    $null
}
if (($null -ne $activePython) -and (Test-Path -LiteralPath $activePython -PathType Leaf)) {
    $launcher = @($activePython)
} else {
    $python = Get-Command python -ErrorAction SilentlyContinue
    $pythonWorks = $false
    if ($null -ne $python) {
        & $python.Source -c "import sys" *> $null
        $pythonWorks = ($LASTEXITCODE -eq 0)
    }
    if ($pythonWorks) {
    $launcher = @($python.Source)
    } else {
        $py = Get-Command py -ErrorAction SilentlyContinue
        if ($null -eq $py) {
            throw "Python 3 is required. Install Python, then rerun this command."
        }
        $launcher = @($py.Source, "-3")
    }
}

if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
    $backendRoot = Join-Path $root ".discovery\native"
} elseif ([System.IO.Path]::IsPathRooted($InstallRoot)) {
    $backendRoot = [System.IO.Path]::GetFullPath($InstallRoot)
} else {
    $backendRoot = [System.IO.Path]::GetFullPath((Join-Path $root $InstallRoot))
}
if (-not [string]::IsNullOrWhiteSpace($Manifest)) {
    $nativeManifest = if ([System.IO.Path]::IsPathRooted($Manifest)) {
        [System.IO.Path]::GetFullPath($Manifest)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path $root $Manifest))
    }
    $arguments += @("--manifest", $nativeManifest)
}
$arguments = @($script) + $arguments + @("--root", $backendRoot)

$executable = $launcher[0]
$prefix = @()
if ($launcher.Count -gt 1) {
    $prefix = $launcher[1..($launcher.Count - 1)]
}
& $executable @prefix @arguments
exit $LASTEXITCODE
