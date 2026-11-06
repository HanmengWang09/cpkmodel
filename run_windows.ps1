[CmdletBinding()]
param(
    [double]$Duration = 8.5,
    [double]$Sample = 0.01,
    [ValidateSet('Radau', 'BDF')][string]$Method = 'Radau',
    [string]$Out = '',
    [string]$Config = '',
    [string]$InitialState = '',
    [switch]$Startup,
    [switch]$NoPlot
)
$ErrorActionPreference = 'Stop'
$taskPython = 'D:\anaconda\envs\biomodel\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw "找不到指定环境：$taskPython" }
$taskRuntime = Join-Path $PSScriptRoot 'runs\runtime'
New-Item -ItemType Directory -Path $taskRuntime -Force | Out-Null
$env:TEMP = $taskRuntime
$env:TMP = $taskRuntime
$env:MPLCONFIGDIR = Join-Path $taskRuntime 'matplotlib'
$env:NUMBA_CACHE_DIR = Join-Path $taskRuntime 'numba'
$env:PIP_CACHE_DIR = Join-Path $taskRuntime 'pip-cache'
$env:PYTHONPYCACHEPREFIX = Join-Path $taskRuntime 'pycache'
$env:PYTHONIOENCODING = 'utf-8'
$taskArguments = @('-m', 'cpkmodel')
if ($Config) { $taskArguments += @('--config', $Config) }
if ($PSBoundParameters.ContainsKey('Duration') -or -not $Config) { $taskArguments += @('--duration', $Duration.ToString([cultureinfo]::InvariantCulture)) }
if ($PSBoundParameters.ContainsKey('Sample') -or -not $Config) { $taskArguments += @('--sample', $Sample.ToString([cultureinfo]::InvariantCulture)) }
if ($PSBoundParameters.ContainsKey('Method') -or -not $Config) { $taskArguments += @('--method', $Method) }
if ($Out) { $taskArguments += @('--out', $Out) }
if ($InitialState) { $taskArguments += @('--initial-state', $InitialState) }
if ($Startup) { $taskArguments += '--startup' }
if ($NoPlot) { $taskArguments += '--no-plot' }
Push-Location -LiteralPath $PSScriptRoot
try {
    & $taskPython @taskArguments
    $taskExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $taskExitCode
