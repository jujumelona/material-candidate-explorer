[CmdletBinding()]
param(
    [string]$Profile = "all-open",
    [string[]]$Component = @(),
    [ValidateSet("auto", "native", "wsl")]
    [string]$Backend = "auto",
    [string]$Manifest,
    [switch]$AllowCustomManifest,
    [string]$InstallRoot,
    [switch]$AllowExternalRoot,
    [switch]$DryRun,
    [ValidateRange(1, 86400)]
    [int]$ReadyTimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"
if (Test-Path -LiteralPath "variable:PSNativeCommandUseErrorActionPreference") {
    $PSNativeCommandUseErrorActionPreference = $false
}

$workspace = [System.IO.Path]::GetFullPath((Split-Path -Parent $MyInvocation.MyCommand.Path))
$defaultManifest = [System.IO.Path]::GetFullPath((Join-Path $workspace "integrations\manifest.v1.json"))

function Get-FullPath {
    param([string]$Value, [string]$Base)
    if ([string]::IsNullOrWhiteSpace($Value)) { throw "A path must not be blank." }
    if ($Value.IndexOfAny([char[]]@([char]0, [char]10, [char]13, [char]9)) -ge 0) {
        throw "Paths must not contain control characters."
    }
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $Base $Value))
}

function Test-PathBelow {
    param([string]$Path, [string]$Parent)
    $fullPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    $fullParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
    if ($fullPath.Equals($fullParent, [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
    $prefix = $fullParent + [System.IO.Path]::DirectorySeparatorChar
    return $fullPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)
}

function Read-Manifest {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Integration manifest does not exist: $Path"
    }
    try {
        $value = Get-Content -Raw -Encoding UTF8 -LiteralPath $Path | ConvertFrom-Json
    } catch {
        throw "Integration manifest is not valid JSON: $($_.Exception.Message)"
    }
    if ($value.schema_version -ne "1.0" -or $null -eq $value.profiles -or $null -eq $value.components) {
        throw "Integration manifest does not satisfy schema version 1.0."
    }
    return $value
}

function Expand-ComponentArguments {
    param([string[]]$Values)
    $result = @()
    foreach ($value in $Values) {
        foreach ($part in ($value -split ',')) {
            $id = $part.Trim()
            if ($id) { $result += $id }
        }
    }
    return @($result | Select-Object -Unique)
}

$componentIds = Expand-ComponentArguments $Component
$manifestPath = if ([string]::IsNullOrWhiteSpace($Manifest)) {
    $defaultManifest
} else {
    Get-FullPath $Manifest $workspace
}
if ((-not $manifestPath.Equals($defaultManifest, [System.StringComparison]::OrdinalIgnoreCase)) -and -not $AllowCustomManifest) {
    throw "Custom manifests are disabled; pass -AllowCustomManifest to trust this file."
}

# Select a backend before creating any state. Linux-only components are never
# silently launched from a native Windows environment.
$manifestValue = Read-Manifest $manifestPath
$backendIds = if ($componentIds.Count -gt 0) {
    $componentIds
} else {
    $profileProperty = $manifestValue.profiles.PSObject.Properties[$Profile]
    if ($null -eq $profileProperty) { throw "Unknown profile '$Profile'." }
    @($profileProperty.Value.components)
}
$requiresLinux = $false
foreach ($id in $backendIds) {
    $entry = @($manifestValue.components | Where-Object { $_.component_id -eq $id })
    if ($entry.Count -ne 1) { throw "Unknown component '$id'." }
    if ($null -ne $entry[0].api -and -not (@($entry[0].platforms) -contains "windows")) {
        $requiresLinux = $true
    }
}
$wsl = Get-Command wsl.exe -ErrorAction SilentlyContinue
$wslUsable = $false
if ($null -ne $wsl) {
    & $wsl.Source --exec sh -c "command -v python3 >/dev/null 2>&1" *> $null
    $wslUsable = ($LASTEXITCODE -eq 0)
}
if ($Backend -eq "auto") {
    if ($requiresLinux -and $wslUsable) { $Backend = "wsl" } else { $Backend = "native" }
}

if ($Backend -eq "wsl") {
    if (-not $wslUsable) {
        throw "-Backend wsl requires a configured Linux distribution with sh and python3."
    }
    $wslWorkspace = (& $wsl.Source --exec wslpath -a $workspace).Trim()
    if ([string]::IsNullOrWhiteSpace($wslWorkspace)) { throw "Could not translate the workspace path into WSL." }
    $wslScript = "$wslWorkspace/start-sidecars.sh"
    if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
        $wslInstallRoot = "$wslWorkspace/.discovery/wsl"
    } elseif ($InstallRoot.StartsWith("/")) {
        $wslInstallRoot = $InstallRoot
    } else {
        $windowsRoot = Get-FullPath $InstallRoot $workspace
        $wslInstallRoot = (& $wsl.Source --exec wslpath -a $windowsRoot).Trim()
        if ([string]::IsNullOrWhiteSpace($wslInstallRoot)) { throw "Could not translate -InstallRoot into WSL." }
    }
    if ($Manifest -and $Manifest.StartsWith("/")) {
        $wslManifest = $Manifest
    } else {
        $wslManifest = (& $wsl.Source --exec wslpath -a $manifestPath).Trim()
        if ([string]::IsNullOrWhiteSpace($wslManifest)) { throw "Could not translate -Manifest into WSL." }
    }
    $arguments = @($wslScript, "--profile", $Profile, "--install-root", $wslInstallRoot,
        "--manifest", $wslManifest, "--ready-timeout-seconds", $ReadyTimeoutSeconds)
    foreach ($id in $componentIds) { $arguments += @("--component", $id) }
    if ($AllowCustomManifest) { $arguments += "--allow-custom-manifest" }
    if ($AllowExternalRoot) { $arguments += "--allow-external-root" }
    if ($DryRun) { $arguments += "--dry-run" }
    $delegatedIds = if ($componentIds.Count -gt 0) {
        $componentIds
    } else {
        $profileProperty = $manifestValue.profiles.PSObject.Properties[$Profile]
        if ($null -eq $profileProperty) { throw "Unknown profile '$Profile'." }
        @($profileProperty.Value.components)
    }
    $bridgeEntries = @()
    foreach ($id in $delegatedIds) {
        $entry = @($manifestValue.components | Where-Object { $_.component_id -eq $id })
        if ($entry.Count -eq 1 -and $null -ne $entry[0].api) {
            $name = ([string]$entry[0].api.base_url_env) -replace '_API_URL$', '_WEIGHT_REVISION'
            if ($name -match '^[A-Z][A-Z0-9_]*_WEIGHT_REVISION$' -and -not [string]::IsNullOrWhiteSpace((Get-Item "Env:$name" -ErrorAction SilentlyContinue).Value)) {
                $bridgeEntries += $name
            }
            $devicePrefix = ([string]$entry[0].api.base_url_env) -replace '_API_URL$', ''
            foreach ($deviceName in @("${devicePrefix}_DEVICE", "${devicePrefix}_CUDA_VISIBLE_DEVICES")) {
                if (-not [string]::IsNullOrWhiteSpace((Get-Item "Env:$deviceName" -ErrorAction SilentlyContinue).Value)) {
                    $bridgeEntries += $deviceName
                }
            }
        }
    }
    foreach ($pathName in @("CHEMPROP_CHECKPOINT_PATH", "REINVENT_MODEL_FILE", "MATTERGEN_CHECKPOINT_PATH",
            "UNIMOL_CHECKPOINT_PATH", "UNIMOL_DICTIONARY_PATH", "MATTERSIM_CHECKPOINT_PATH", "CHGNET_CHECKPOINT_PATH",
            "SCGPT_CHECKPOINT_DIR", "QHNET_CHECKPOINT_PATH", "QHNET_CONFIG_PATH", "BOLTZ_CACHE")) {
        if (-not [string]::IsNullOrWhiteSpace((Get-Item "Env:$pathName" -ErrorAction SilentlyContinue).Value)) {
            $bridgeEntries += "$pathName/p"
        }
    }
    foreach ($configName in @("SIDECAR_DEVICE", "MATTERGEN_PRETRAINED_NAME", "MATTERGEN_OBJECTIVE_MAP",
            "REINVENT_MODE", "UMA_MODEL_NAME", "UMA_TASK_NAME", "CHGNET_MODEL_NAME",
            "CHEMPROP_PROPERTY_NAMES", "CHEMPROP_PROPERTY_UNITS", "CHEMPROP_ENCODING_LAYER", "ESM_MODEL_NAME", "PYSCF_BASIS",
            "SCGPT_MAX_GENES", "SCGPT_MAX_LENGTH", "SCGPT_USE_FAST_TRANSFORMER",
            "BOLTZ_PROCESS_TIMEOUT_SECONDS", "BOLTZ_MAX_JSON_BYTES", "BOLTZ_MAX_CIF_BYTES",
            "BOLTZ_MAX_SEQUENCE_LENGTH", "BOLTZ_MAX_SMILES_LENGTH", "BOLTZ_NO_KERNELS")) {
        if (-not [string]::IsNullOrWhiteSpace((Get-Item "Env:$configName" -ErrorAction SilentlyContinue).Value)) {
            $bridgeEntries += $configName
        }
    }
    $savedWslenv = $env:WSLENV
    try {
        $combined = @($savedWslenv, $bridgeEntries) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
        if ($combined.Count -gt 0) { $env:WSLENV = $combined -join ':' }
        & $wsl.Source --exec sh @arguments
        $wslExitCode = $LASTEXITCODE
    } finally {
        if ($null -eq $savedWslenv) { Remove-Item Env:WSLENV -ErrorAction SilentlyContinue }
        else { $env:WSLENV = $savedWslenv }
    }
    exit $wslExitCode
}

$resolvedInstallRoot = if ([string]::IsNullOrWhiteSpace($InstallRoot)) {
    [System.IO.Path]::GetFullPath((Join-Path $workspace ".discovery\native"))
} else {
    Get-FullPath $InstallRoot $workspace
}
$volumeRoot = [System.IO.Path]::GetPathRoot($resolvedInstallRoot).TrimEnd('\', '/')
if ($resolvedInstallRoot.TrimEnd('\', '/').Equals($volumeRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "InstallRoot cannot be a filesystem root."
}
if ((-not $AllowExternalRoot) -and (-not (Test-PathBelow $resolvedInstallRoot $workspace))) {
    throw "InstallRoot is outside the workspace; pass -AllowExternalRoot to use it explicitly."
}

function Assert-SafeSlug {
    param([string]$Value, [string]$Label)
    if ($Value -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$') { throw "$Label is unsafe: '$Value'." }
}

function Assert-SafeMetadata {
    param([string]$Value, [string]$Label)
    if ([string]::IsNullOrWhiteSpace($Value) -or $Value -notmatch '^[A-Za-z0-9][A-Za-z0-9._:+/@=-]{0,255}$') {
        throw "$Label contains unsupported characters or is blank."
    }
}

function Get-EnvironmentValue {
    param([string]$Name)
    return [System.Environment]::GetEnvironmentVariable($Name, "Process")
}

function Get-SensitiveEnvironmentNames {
    return @(Get-ChildItem Env: | Where-Object {
        $_.Name -match '(?i)(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_KEY|API_KEY|ACCESS_KEY)' -or
        $_.Name -in @('SSH_AUTH_SOCK', 'GPG_AGENT_INFO')
    } | ForEach-Object { $_.Name })
}

function Get-WeightRevision {
    param($Entry)
    $prefix = $Entry.api.base_url_env -replace '_API_URL$', ''
    $envName = "${prefix}_WEIGHT_REVISION"
    $weights = @($Entry.weights)
    $unresolved = @($weights | Where-Object { $_.kind -in @("managed", "manual") -or [string]::IsNullOrWhiteSpace($_.revision) })
    if ($unresolved.Count -gt 0) {
        $configured = Get-EnvironmentValue $envName
        if ([string]::IsNullOrWhiteSpace($configured)) {
            throw "$envName is required because '$($Entry.component_id)' uses managed/manual or ambiguous weights."
        }
        Assert-SafeMetadata $configured $envName
        return [PSCustomObject]@{ revision = $configured.Trim(); env_name = $envName }
    }
    if ($weights.Count -eq 0) {
        return [PSCustomObject]@{ revision = "no-external-weight"; env_name = $envName }
    }
    $revisions = @($weights | ForEach-Object { $_.revision } | Select-Object -Unique)
    if ($revisions.Count -ne 1) { throw "$envName is required because weight revisions are ambiguous." }
    Assert-SafeMetadata $revisions[0] "$($Entry.component_id) weight revision"
    return [PSCustomObject]@{ revision = $revisions[0]; env_name = $envName }
}

function Get-FileAttestation {
    param([string]$Path, [string]$Declared, [string]$Label)
    $selected = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if ($selected.PSIsContainer -or ($selected.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
        throw "$Label must be a regular non-symlink file."
    }
    $resolved = (Resolve-Path -LiteralPath $selected.FullName -ErrorAction Stop).Path
    if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { throw "$Label must be a file." }
    $actual = "sha256:" + (Get-FileHash -LiteralPath $resolved -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not [string]::IsNullOrWhiteSpace($Declared) -and $Declared.StartsWith("sha256:", [System.StringComparison]::OrdinalIgnoreCase) -and -not $Declared.Equals($actual, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label declared '$Declared' but the selected file is $actual."
    }
    return [PSCustomObject]@{ path = $resolved; revision = $actual }
}

function Get-DirectoryInventoryAttestation {
    param([string]$Path, [string]$Declared, [string]$Label)
    $rootItem = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
    if (-not $rootItem.PSIsContainer -or ($rootItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
        throw "$Label must be a regular non-symlink directory."
    }
    $resolved = $rootItem.FullName.TrimEnd('\', '/')
    $prefix = $resolved + [System.IO.Path]::DirectorySeparatorChar
    $relativeToPath = @{}
    foreach ($item in @(Get-ChildItem -LiteralPath $resolved -Recurse -Force)) {
        if (-not $item.FullName.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "$Label inventory escapes its root."
        }
        $relativeItem = $item.FullName.Substring($prefix.Length).Replace('\', '/')
        $parts = @($relativeItem -split '/')
        if ($parts.Count -gt 1 -and (@($parts[0..($parts.Count - 2)]) -contains ".cache")) {
            continue
        }
        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            throw "$Label inventory contains a symlink: $($item.FullName)"
        }
        if (-not $item.PSIsContainer -and $item.Name -ne ".snapshot.json") {
            $relativeToPath[$relativeItem] = $item.FullName
        }
    }
    if ($relativeToPath.Count -eq 0) { throw "$Label contains no checkpoint files." }
    [string[]]$relativeNames = @($relativeToPath.Keys)
    [Array]::Sort($relativeNames, [System.StringComparer]::Ordinal)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        foreach ($relative in $relativeNames) {
            $nameBytes = [System.Text.Encoding]::UTF8.GetBytes($relative)
            [byte[]]$nameLength = [System.BitConverter]::GetBytes([uint32]$nameBytes.Length)
            [byte[]]$fileLength = [System.BitConverter]::GetBytes([uint64](Get-Item -LiteralPath $relativeToPath[$relative]).Length)
            if ([System.BitConverter]::IsLittleEndian) {
                [Array]::Reverse($nameLength)
                [Array]::Reverse($fileLength)
            }
            [void]$sha.TransformBlock($nameLength, 0, $nameLength.Length, $nameLength, 0)
            [void]$sha.TransformBlock($nameBytes, 0, $nameBytes.Length, $nameBytes, 0)
            [void]$sha.TransformBlock($fileLength, 0, $fileLength.Length, $fileLength, 0)
            $stream = [System.IO.File]::OpenRead($relativeToPath[$relative])
            try {
                [byte[]]$buffer = New-Object byte[] (1024 * 1024)
                while (($count = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                    [void]$sha.TransformBlock($buffer, 0, $count, $buffer, 0)
                }
            } finally {
                $stream.Dispose()
            }
        }
        [void]$sha.TransformFinalBlock([byte[]]@(), 0, 0)
        $digest = ([System.BitConverter]::ToString($sha.Hash)).Replace("-", "").ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
    $actual = "sha256:$digest"
    if (-not [string]::IsNullOrWhiteSpace($Declared) -and $Declared.StartsWith("sha256:", [System.StringComparison]::OrdinalIgnoreCase) -and -not $Declared.Equals($actual, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label declared '$Declared' but the selected directory is $actual."
    }
    return [PSCustomObject]@{ path = $resolved; revision = $actual; digest = $digest }
}

function Get-WeightBinding {
    param($Entry, [string]$Root)
    $prefix = $Entry.api.base_url_env -replace '_API_URL$', ''
    $envName = "${prefix}_WEIGHT_REVISION"
    $weights = @($Entry.weights)
    $runtime = [ordered]@{}
    if ($weights.Count -eq 0) {
        return [PSCustomObject]@{ revision = "no-external-weight"; env_name = $envName; runtime = $runtime }
    }
    if ($weights.Count -ne 1) { throw "$envName is required because weight revisions are ambiguous." }
    $item = $weights[0]
    $declared = Get-EnvironmentValue $envName
    if ($item.kind -eq "huggingface") {
        $snapshot = [System.IO.Path]::GetFullPath((Join-Path $Root "models\$($Entry.component_id)\$($item.weight_id)\$($item.revision)"))
        if (-not (Test-PathBelow $snapshot $Root)) { throw "Weight snapshot escapes InstallRoot." }
        $markerPath = Join-Path $snapshot ".snapshot.json"
        if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) { throw "Verified snapshot marker is missing: $markerPath" }
        try { $marker = Get-Content -Raw -Encoding UTF8 -LiteralPath $markerPath | ConvertFrom-Json }
        catch { throw "Snapshot marker is unreadable: $markerPath" }
        if ($marker.schema_version -ne "1.0" -or $marker.repository -ne $item.repository -or $marker.revision -ne $item.revision) {
            throw "Snapshot marker repository/revision does not match the integration manifest for '$($Entry.component_id)'."
        }
        if ([string]$marker.inventory_sha256 -notmatch '^[0-9a-f]{64}$') {
            throw "Snapshot marker inventory_sha256 is missing or invalid for '$($Entry.component_id)'."
        }
        $runtime["SIDECAR_WEIGHT_SNAPSHOT_PATH"] = $snapshot
        $runtime["SIDECAR_WEIGHT_ATTESTATION"] = "huggingface:$($item.repository)@$($item.revision)"
        return [PSCustomObject]@{ revision = [string]$item.revision; env_name = $envName; runtime = $runtime }
    }
    if ($item.kind -eq "https") {
        $downloadUri = [System.Uri]([string]$item.download_url)
        $filename = [System.IO.Path]::GetFileName($downloadUri.AbsolutePath)
        if ($filename -notmatch '^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,254}[A-Za-z0-9])?$') {
            throw "HTTPS weight for '$($Entry.component_id)' has an unsafe filename."
        }
        $artifactRoot = [System.IO.Path]::GetFullPath(
            (Join-Path $Root "models\$($Entry.component_id)\$($item.weight_id)\$($item.revision)")
        )
        $artifact = [System.IO.Path]::GetFullPath((Join-Path $artifactRoot $filename))
        if (-not (Test-PathBelow $artifactRoot $Root) -or -not (Test-PathBelow $artifact $artifactRoot)) {
            throw "HTTPS weight artifact escapes InstallRoot."
        }
        $markerPath = Join-Path $artifactRoot ".artifact.json"
        if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
            throw "Verified artifact marker is missing: $markerPath"
        }
        try { $marker = Get-Content -Raw -Encoding UTF8 -LiteralPath $markerPath | ConvertFrom-Json }
        catch { throw "Artifact marker is unreadable: $markerPath" }
        if (
            $marker.schema_version -ne "1.0" -or
            $marker.download_url -ne $item.download_url -or
            $marker.revision -ne $item.revision -or
            $marker.filename -ne $filename -or
            $marker.sha256 -ne $item.sha256 -or
            [long]$marker.size_bytes -ne [long]$item.expected_size_bytes
        ) {
            throw "Artifact marker does not match the integration manifest for '$($Entry.component_id)'."
        }
        $attested = Get-FileAttestation $artifact "" $filename
        $expectedRevision = "sha256:$($item.sha256)"
        if (
            $attested.revision -ne $expectedRevision -or
            (Get-Item -LiteralPath $artifact).Length -ne [long]$item.expected_size_bytes
        ) {
            throw "Verified HTTPS weight bytes changed for '$($Entry.component_id)'."
        }
        if (
            -not [string]::IsNullOrWhiteSpace($declared) -and
            -not $declared.Equals($expectedRevision, [System.StringComparison]::OrdinalIgnoreCase)
        ) {
            throw "$envName conflicts with the manifest-pinned HTTPS weight."
        }
        $checkpointName = if ($Entry.component_id -eq "mattersim") {
            "MATTERSIM_CHECKPOINT_PATH"
        } elseif ($Entry.component_id -eq "chgnet") {
            "CHGNET_CHECKPOINT_PATH"
        } else {
            $null
        }
        if ($null -eq $checkpointName) {
            throw "HTTPS weight binding is not implemented for '$($Entry.component_id)'."
        }
        $runtime[$checkpointName] = $attested.path
        $runtime["SIDECAR_WEIGHT_ATTESTATION"] = $expectedRevision
        return [PSCustomObject]@{ revision = $expectedRevision; env_name = $envName; runtime = $runtime }
    }
    if ($item.kind -eq "manual") {
        if ($Entry.component_id -eq "qhnet-source") {
            $checkpointRaw = Get-EnvironmentValue "QHNET_CHECKPOINT_PATH"
            $configRaw = Get-EnvironmentValue "QHNET_CONFIG_PATH"
            if ([string]::IsNullOrWhiteSpace($checkpointRaw)) { throw "QHNET_CHECKPOINT_PATH is required to start the QHNet sidecar." }
            if ([string]::IsNullOrWhiteSpace($configRaw)) { throw "QHNET_CONFIG_PATH is required to start the QHNet sidecar." }
            $checkpoint = Get-FileAttestation $checkpointRaw "" "QHNET_CHECKPOINT_PATH"
            $config = Get-FileAttestation $configRaw "" "QHNET_CONFIG_PATH"
            $material = '{"checkpoint_sha256":"' + $checkpoint.revision.Substring(7) + '","config_sha256":"' + $config.revision.Substring(7) + '"}'
            $sha = [System.Security.Cryptography.SHA256]::Create()
            try {
                $digestBytes = $sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($material))
                $digest = ([System.BitConverter]::ToString($digestBytes)).Replace("-", "").ToLowerInvariant()
            } finally { $sha.Dispose() }
            $revision = "bundle-sha256:$digest"
            if (-not [string]::IsNullOrWhiteSpace($declared) -and -not $declared.Equals($revision, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "QHNET_WEIGHT_REVISION conflicts with the selected checkpoint/config bundle."
            }
            $sourceRoot = [System.IO.Path]::GetFullPath((Join-Path $Root "sources\qhnet-source"))
            if (-not (Test-PathBelow $sourceRoot $Root) -or -not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
                throw "Verified QHNet source was not found under InstallRoot; run bootstrap first."
            }
            $runtime["QHNET_SOURCE_PATH"] = $sourceRoot
            $runtime["QHNET_CHECKPOINT_PATH"] = $checkpoint.path
            $runtime["QHNET_CONFIG_PATH"] = $config.path
            $runtime["SIDECAR_WEIGHT_ATTESTATION"] = $revision
            return [PSCustomObject]@{ revision = $revision; env_name = $envName; runtime = $runtime }
        }
        if ($Entry.component_id -eq "scgpt") {
            $rawPath = Get-EnvironmentValue "SCGPT_CHECKPOINT_DIR"
            if ([string]::IsNullOrWhiteSpace($rawPath)) { throw "SCGPT_CHECKPOINT_DIR is required to start the scgpt sidecar." }
            $attested = Get-DirectoryInventoryAttestation $rawPath $declared "SCGPT_CHECKPOINT_DIR"
            foreach ($name in @("args.json", "vocab.json", "best_model.pt")) {
                $member = Join-Path $attested.path $name
                $memberItem = Get-Item -LiteralPath $member -Force -ErrorAction Stop
                if ($memberItem.PSIsContainer -or ($memberItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
                    throw "SCGPT_CHECKPOINT_DIR requires regular non-symlink file '$name'."
                }
            }
            $runtime["SCGPT_CHECKPOINT_DIR"] = $attested.path
            $runtime["SCGPT_BUNDLE_INVENTORY_SHA256"] = $attested.digest
            $runtime["SIDECAR_WEIGHT_ATTESTATION"] = $attested.revision
            return [PSCustomObject]@{ revision = $attested.revision; env_name = $envName; runtime = $runtime }
        }
        $pathName = if ($Entry.component_id -eq "chemprop") { "CHEMPROP_CHECKPOINT_PATH" } elseif ($Entry.component_id -eq "reinvent4") { "REINVENT_MODEL_FILE" } else { $null }
        if ($null -eq $pathName) { throw "Manual weight binding is not implemented for '$($Entry.component_id)'." }
        $rawPath = Get-EnvironmentValue $pathName
        if ([string]::IsNullOrWhiteSpace($rawPath)) { throw "$pathName is required to start the $($Entry.component_id) sidecar." }
        $attested = Get-FileAttestation $rawPath $declared $pathName
        $runtime[$pathName] = $attested.path
        $runtime["SIDECAR_WEIGHT_ATTESTATION"] = $attested.revision
        return [PSCustomObject]@{ revision = $attested.revision; env_name = $envName; runtime = $runtime }
    }
    if ($item.kind -eq "managed") {
        $managedPathName = if ($Entry.component_id -eq "mattersim") {
            "MATTERSIM_CHECKPOINT_PATH"
        } elseif ($Entry.component_id -eq "chgnet") {
            "CHGNET_CHECKPOINT_PATH"
        } else {
            $null
        }
        if ($null -ne $managedPathName) {
            $localPath = Get-EnvironmentValue $managedPathName
            if (-not [string]::IsNullOrWhiteSpace($localPath)) {
                $attested = Get-FileAttestation $localPath $declared $managedPathName
                $runtime[$managedPathName] = $attested.path
                $runtime["SIDECAR_WEIGHT_ATTESTATION"] = $attested.revision
                return [PSCustomObject]@{ revision = $attested.revision; env_name = $envName; runtime = $runtime }
            }
        }
        if ([string]::IsNullOrWhiteSpace($declared)) { throw "$envName is required because '$($Entry.component_id)' uses managed weights." }
        if ($declared.StartsWith("sha256:", [System.StringComparison]::OrdinalIgnoreCase)) { throw "$envName cannot claim SHA-256 without a selected local checkpoint file." }
        $effective = if ($declared.StartsWith("managed-unattested:")) { $declared } else { "managed-unattested:$declared" }
        Assert-SafeMetadata $effective $envName
        $runtime["SIDECAR_WEIGHT_ATTESTATION"] = $effective
        return [PSCustomObject]@{ revision = $effective; env_name = $envName; runtime = $runtime }
    }
    throw "Unsupported weight kind '$($item.kind)'."
}

$selectedIds = if ($componentIds.Count -gt 0) {
    $componentIds
} else {
    $property = $manifestValue.profiles.PSObject.Properties[$Profile]
    if ($null -eq $property) { throw "Unknown profile '$Profile'." }
    @($property.Value.components)
}
if ($selectedIds.Count -eq 0) { throw "No components were selected." }

$plans = @()
foreach ($id in $selectedIds) {
    Assert-SafeSlug $id "component_id"
    $matches = @($manifestValue.components | Where-Object { $_.component_id -eq $id })
    if ($matches.Count -ne 1) { throw "Unknown or duplicate component '$id'." }
    $entry = $matches[0]
    if ($null -eq $entry.api) { continue }
    if (-not (@($entry.platforms) -contains "windows")) {
        throw "Component '$id' does not support native Windows; use -Backend wsl."
    }
    if ($entry.api.base_url_env -notmatch '^[A-Z][A-Z0-9_]*_API_URL$') {
        throw "Component '$id' has an unsafe API environment name."
    }
    $port = [int]$entry.api.default_port
    if ($port -lt 1 -or $port -gt 65535) { throw "Component '$id' has an invalid port." }
    $version = if (-not [string]::IsNullOrWhiteSpace($entry.install.version)) {
        [string]$entry.install.version
    } elseif ($null -ne $entry.source -and -not [string]::IsNullOrWhiteSpace($entry.source.release)) {
        [string]$entry.source.release
    } else { "remote" }
    $codeRevision = if ($null -ne $entry.source) { [string]$entry.source.revision } else { "workspace" }
    Assert-SafeMetadata $version "$id model version"
    Assert-SafeMetadata $codeRevision "$id code revision"
    $weight = Get-WeightBinding $entry $resolvedInstallRoot
    $runtimePrefix = ([string]$entry.api.base_url_env) -replace '_API_URL$', ''
    $componentDevice = Get-EnvironmentValue "${runtimePrefix}_DEVICE"
    if ([string]::IsNullOrWhiteSpace($componentDevice)) {
        $componentDevice = Get-EnvironmentValue "SIDECAR_DEVICE"
    }
    if (-not [string]::IsNullOrWhiteSpace($componentDevice)) {
        $weight.runtime["SIDECAR_DEVICE"] = $componentDevice.Trim()
    }
    $componentVisibleDevices = Get-EnvironmentValue "${runtimePrefix}_CUDA_VISIBLE_DEVICES"
    if (-not [string]::IsNullOrWhiteSpace($componentVisibleDevices)) {
        if ($componentVisibleDevices.IndexOfAny([char[]]@([char]0, [char]10, [char]13, [char]9)) -ge 0) {
            throw "${runtimePrefix}_CUDA_VISIBLE_DEVICES contains control characters."
        }
        $weight.runtime["CUDA_VISIBLE_DEVICES"] = $componentVisibleDevices.Trim()
    }
    $allowedConfigurationNames = @(
        "SIDECAR_MAX_REQUEST_BYTES", "SIDECAR_MAX_BATCH_SIZE",
        "SIDECAR_MAX_CONCURRENCY", "SIDECAR_MAX_QUEUE_SIZE", "SIDECAR_TIMEOUT_SECONDS",
        "MATTERGEN_PRETRAINED_NAME", "MATTERGEN_OBJECTIVE_MAP", "UNIMOL_REMOVE_HS",
        "REINVENT_MODE", "UMA_MODEL_NAME", "UMA_TASK_NAME", "CHGNET_MODEL_NAME",
        "CHEMPROP_PROPERTY_NAMES", "CHEMPROP_PROPERTY_UNITS", "CHEMPROP_ENCODING_LAYER", "ESM_MODEL_NAME", "PYSCF_BASIS",
        "SCGPT_MAX_GENES", "SCGPT_MAX_LENGTH", "SCGPT_USE_FAST_TRANSFORMER", "BOLTZ_CACHE",
        "BOLTZ_PROCESS_TIMEOUT_SECONDS", "BOLTZ_MAX_JSON_BYTES", "BOLTZ_MAX_CIF_BYTES",
        "BOLTZ_MAX_SEQUENCE_LENGTH", "BOLTZ_MAX_SMILES_LENGTH", "BOLTZ_NO_KERNELS"
    )
    foreach ($configurationName in $allowedConfigurationNames) {
        $configurationValue = Get-EnvironmentValue $configurationName
        if (-not [string]::IsNullOrWhiteSpace($configurationValue)) {
            $weight.runtime[$configurationName] = $configurationValue
        }
    }
    $environment = [System.IO.Path]::GetFullPath((Join-Path $resolvedInstallRoot "envs\$id"))
    $python = [System.IO.Path]::GetFullPath((Join-Path $environment "Scripts\python.exe"))
    if (-not (Test-PathBelow $python $resolvedInstallRoot)) { throw "Environment for '$id' escapes InstallRoot." }
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        throw "Installed Python for '$id' was not found at $python. Run bootstrap for this component first."
    }
    $resolvedPython = (Resolve-Path -LiteralPath $python).Path
    if (-not (Test-PathBelow $resolvedPython $resolvedInstallRoot)) { throw "Python for '$id' resolves outside InstallRoot." }
    if ($id -eq "chemprop" -and [string]::IsNullOrWhiteSpace((Get-EnvironmentValue "CHEMPROP_CHECKPOINT_PATH"))) {
        throw "CHEMPROP_CHECKPOINT_PATH is required to start the Chemprop sidecar."
    }
    if ($id -eq "chemprop" -and [string]::IsNullOrWhiteSpace((Get-EnvironmentValue "CHEMPROP_PROPERTY_NAMES"))) {
        throw "CHEMPROP_PROPERTY_NAMES is required to start the Chemprop sidecar."
    }
    if ($id -eq "chemprop" -and [string]::IsNullOrWhiteSpace((Get-EnvironmentValue "CHEMPROP_PROPERTY_UNITS"))) {
        throw "CHEMPROP_PROPERTY_UNITS is required to start the Chemprop sidecar."
    }
    if ($id -eq "reinvent4" -and [string]::IsNullOrWhiteSpace((Get-EnvironmentValue "REINVENT_MODEL_FILE"))) {
        throw "REINVENT_MODEL_FILE is required to start the REINVENT sidecar."
    }
    $cliId = if ($id -eq "qhnet-source") { "qhnet" } else { $id }
    $url = "http://127.0.0.1:$port"
    $statePath = Join-Path $resolvedInstallRoot "state\sidecars\$id.json"
    $plans += [PSCustomObject]@{
        component_id = $id
        cli_model_id = $cliId
        python = $resolvedPython
        port = $port
        api_env = [string]$entry.api.base_url_env
        url = $url
        model_version = $version
        code_revision = $codeRevision
        weight_revision = $weight.revision
        weight_env = $weight.env_name
        runtime_environment = $weight.runtime
        state_path = $statePath
        command = @($resolvedPython, "-m", "discovery_os.sidecars", "--model", $cliId, "--host", "127.0.0.1", "--port", "$port")
    }
}
if ($plans.Count -eq 0) { throw "The selected profile has no launchable API sidecars." }
$duplicatePorts = @($plans | Group-Object port | Where-Object { $_.Count -gt 1 })
if ($duplicatePorts.Count -gt 0) { throw "Selected sidecars contain duplicate loopback ports." }
$duplicateApiVariables = @($plans | Group-Object api_env | Where-Object { $_.Count -gt 1 })
if ($duplicateApiVariables.Count -gt 0) { throw "Selected sidecars contain duplicate API environment variables." }

function Test-TrackedProcessAlive {
    param([string]$StatePath)
    if (-not (Test-Path -LiteralPath $StatePath -PathType Leaf)) { return $false }
    try {
        $state = Get-Content -Raw -Encoding UTF8 -LiteralPath $StatePath | ConvertFrom-Json
        $pidValue = [int]$state.pid
        if ($pidValue -le 0) { return $false }
        return $null -ne (Get-Process -Id $pidValue -ErrorAction SilentlyContinue)
    } catch { throw "Refusing to replace unreadable sidecar state: $StatePath" }
}

function Test-LoopbackPortOpen {
    param([int]$Port)
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $pending = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if (-not $pending.AsyncWaitHandle.WaitOne(250)) { return $false }
        $client.EndConnect($pending)
        return $true
    } catch { return $false } finally { $client.Close() }
}

if (-not $DryRun) {
    foreach ($plan in $plans) {
        if (Test-TrackedProcessAlive $plan.state_path) {
            throw "Refusing to overwrite live sidecar process state for '$($plan.component_id)'."
        }
        if (Test-LoopbackPortOpen $plan.port) {
            throw "Refusing to start '$($plan.component_id)': loopback port $($plan.port) is already in use."
        }
    }
    # Transactional configuration preflight: every isolated environment must
    # confirm adapter support and static configuration before the first web
    # server process is started. This never loads a model checkpoint.
    foreach ($plan in $plans) {
        $saved = @{}
        $bindingNames = @($plan.runtime_environment.Keys)
        $secretNames = Get-SensitiveEnvironmentNames
        foreach ($name in @(@("SIDECAR_MODEL_VERSION", "SIDECAR_CODE_REVISION", "SIDECAR_WEIGHT_REVISION") + $bindingNames + $secretNames | Select-Object -Unique)) {
            $saved[$name] = Get-EnvironmentValue $name
        }
        $preflightExitCode = -1
        $preflightOutput = @()
        try {
            foreach ($name in $secretNames) { Remove-Item "Env:$name" -ErrorAction SilentlyContinue }
            $env:SIDECAR_MODEL_VERSION = $plan.model_version
            $env:SIDECAR_CODE_REVISION = $plan.code_revision
            $env:SIDECAR_WEIGHT_REVISION = $plan.weight_revision
            foreach ($name in $bindingNames) {
                [System.Environment]::SetEnvironmentVariable($name, [string]$plan.runtime_environment[$name], "Process")
            }
            $preflightOutput = @(& $plan.python -m discovery_os.sidecars --model $plan.cli_model_id `
                --host 127.0.0.1 --port $plan.port --preflight 2>&1)
            $preflightExitCode = $LASTEXITCODE
        } finally {
            foreach ($name in $saved.Keys) {
                if ($null -eq $saved[$name]) { Remove-Item "Env:$name" -ErrorAction SilentlyContinue }
                else { [System.Environment]::SetEnvironmentVariable($name, [string]$saved[$name], "Process") }
            }
        }
        if ($preflightExitCode -ne 0) {
            $detail = ($preflightOutput | Out-String).Trim()
            throw "Sidecar configuration preflight failed for '$($plan.component_id)' before any server was started: $detail"
        }
        try { $preflight = ($preflightOutput | Out-String) | ConvertFrom-Json }
        catch { throw "Sidecar preflight returned invalid JSON for '$($plan.component_id)'." }
        if ([string]::IsNullOrWhiteSpace([string]$preflight.runtime_parameters_hash) -or [string]$preflight.runtime_parameters_hash -notmatch '^[0-9a-f]{64}$') {
            throw "Sidecar preflight omitted a valid runtime_parameters_hash for '$($plan.component_id)'."
        }
        $plan | Add-Member -NotePropertyName runtime_parameters_hash -NotePropertyValue ([string]$preflight.runtime_parameters_hash) -Force
    }
}

$stateDirectory = Join-Path $resolvedInstallRoot "state\sidecars"
$logDirectory = Join-Path $resolvedInstallRoot "logs\sidecars"
New-Item -ItemType Directory -Force -Path $stateDirectory, $logDirectory | Out-Null
if (-not (Test-PathBelow (Resolve-Path $stateDirectory).Path $resolvedInstallRoot)) { throw "State directory escapes InstallRoot." }
if (-not (Test-PathBelow (Resolve-Path $logDirectory).Path $resolvedInstallRoot)) { throw "Log directory escapes InstallRoot." }

function Write-AtomicUtf8 {
    param([string]$Path, [string[]]$Lines)
    $temporary = "$Path.$PID.tmp"
    [System.IO.File]::WriteAllLines($temporary, $Lines, (New-Object System.Text.UTF8Encoding($false)))
    Move-Item -Force -LiteralPath $temporary -Destination $Path
}

function Quote-PowerShellLiteral { param([string]$Value); return "'" + $Value.Replace("'", "''") + "'" }
function Quote-ShLiteral {
    param([string]$Value)
    $embeddedQuote = "'" + [char]92 + "''"
    return "'" + $Value.Replace("'", $embeddedQuote) + "'"
}

$psEnvironment = @("# Generated by start-sidecars.ps1. Contains endpoints and revisions only; no credentials.",
    "`$env:DISCOVERY_ALLOW_INSECURE_LOCAL_HTTP = '1'")
$shEnvironment = @("# Generated by start-sidecars.ps1. Contains endpoints and revisions only; no credentials.",
    "export DISCOVERY_ALLOW_INSECURE_LOCAL_HTTP='1'")
foreach ($plan in $plans) {
    $psEnvironment += "`$env:$($plan.api_env) = $(Quote-PowerShellLiteral $plan.url)"
    $psEnvironment += "`$env:$($plan.weight_env) = $(Quote-PowerShellLiteral $plan.weight_revision)"
    $shEnvironment += "export $($plan.api_env)=$(Quote-ShLiteral $plan.url)"
    $shEnvironment += "export $($plan.weight_env)=$(Quote-ShLiteral $plan.weight_revision)"
    if (-not [string]::IsNullOrWhiteSpace([string]$plan.runtime_parameters_hash)) {
        $runtimeHashEnv = $plan.api_env -replace '_API_URL$', '_RUNTIME_PARAMETERS_HASH'
        $psEnvironment += "`$env:$runtimeHashEnv = $(Quote-PowerShellLiteral $plan.runtime_parameters_hash)"
        $shEnvironment += "export $runtimeHashEnv=$(Quote-ShLiteral $plan.runtime_parameters_hash)"
    }
}
$envPs1 = Join-Path $resolvedInstallRoot "sidecars.env.ps1"
$envSh = Join-Path $resolvedInstallRoot "sidecars.env.sh"
Write-AtomicUtf8 $envPs1 $psEnvironment
Write-AtomicUtf8 $envSh $shEnvironment

if ($DryRun) {
    [PSCustomObject]@{
        schema_version = "1.0"
        dry_run = $true
        profile = $Profile
        install_root = $resolvedInstallRoot
        manifest_revision = $manifestValue.manifest_revision
        env_files = @($envPs1, $envSh)
        sidecars = $plans
    } | ConvertTo-Json -Depth 8
    exit 0
}

function Write-State {
    param($Plan, [int]$ProcessId, [string]$Status, [string]$StdoutLog, [string]$StderrLog)
    $payload = [ordered]@{
        schema_version = "1.0"; component_id = $Plan.component_id; pid = $ProcessId
        status = $Status; url = $Plan.url; command = $Plan.command
        model_version = $Plan.model_version; code_revision = $Plan.code_revision
        weight_revision = $Plan.weight_revision; runtime_parameters_hash = $Plan.runtime_parameters_hash
        stdout_log = $StdoutLog; stderr_log = $StderrLog
        updated_at = [DateTimeOffset]::UtcNow.ToString("o")
    } | ConvertTo-Json -Depth 6
    Write-AtomicUtf8 $Plan.state_path @($payload)
}

foreach ($plan in $plans) {
    $stamp = [DateTimeOffset]::UtcNow.ToString("yyyyMMddTHHmmssfffZ")
    $stdoutLog = Join-Path $logDirectory "$($plan.component_id)-$stamp.out.log"
    $stderrLog = Join-Path $logDirectory "$($plan.component_id)-$stamp.err.log"
    $saved = @{}
    $bindingNames = @($plan.runtime_environment.Keys)
    $secretNames = Get-SensitiveEnvironmentNames
    foreach ($name in @(@("SIDECAR_MODEL_VERSION", "SIDECAR_CODE_REVISION", "SIDECAR_WEIGHT_REVISION", "PYTHONUNBUFFERED") + $bindingNames + $secretNames | Select-Object -Unique)) {
        $saved[$name] = Get-EnvironmentValue $name
    }
    try {
        foreach ($name in $secretNames) { Remove-Item "Env:$name" -ErrorAction SilentlyContinue }
        $env:SIDECAR_MODEL_VERSION = $plan.model_version
        $env:SIDECAR_CODE_REVISION = $plan.code_revision
        $env:SIDECAR_WEIGHT_REVISION = $plan.weight_revision
        $env:PYTHONUNBUFFERED = "1"
        foreach ($name in $bindingNames) {
            [System.Environment]::SetEnvironmentVariable($name, [string]$plan.runtime_environment[$name], "Process")
        }
        $process = Start-Process -FilePath $plan.python -ArgumentList @(
            "-m", "discovery_os.sidecars", "--model", $plan.cli_model_id,
            "--host", "127.0.0.1", "--port", "$($plan.port)"
        ) -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
    } finally {
        foreach ($name in $saved.Keys) {
            if ($null -eq $saved[$name]) { Remove-Item "Env:$name" -ErrorAction SilentlyContinue }
            else { [System.Environment]::SetEnvironmentVariable($name, [string]$saved[$name], "Process") }
        }
    }
    Write-State $plan $process.Id "starting" $stdoutLog $stderrLog
    $deadline = [DateTimeOffset]::UtcNow.AddSeconds($ReadyTimeoutSeconds)
    $ready = $false
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        if ($process.HasExited) { break }
        try {
            $health = Invoke-RestMethod -UseBasicParsing -Uri "$($plan.url)/health" -TimeoutSec 2
            if ($health.ready -eq $true) { $ready = $true; break }
        } catch { }
        Start-Sleep -Milliseconds 500
        $process.Refresh()
    }
    if (-not $ready) {
        $status = if ($process.HasExited) { "exited" } else { "readiness_timeout" }
        Write-State $plan $process.Id $status $stdoutLog $stderrLog
        throw "Sidecar '$($plan.component_id)' did not become ready within $ReadyTimeoutSeconds seconds. Process was left untouched; inspect $stderrLog"
    }
    Write-State $plan $process.Id "ready" $stdoutLog $stderrLog
    Write-Host "[sidecar] ready $($plan.component_id) pid=$($process.Id) $($plan.url)"
}

Write-Host "Source $envPs1 to configure the central Fusion Core clients."
