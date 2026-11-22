param (
    [Parameter(Mandatory=$false)]
    [ValidateSet("start", "stop", "status", "restart")]
    [string]$Action = "status"
)

$PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Strategies = @(
    @{ Id="01"; Name="BTCSwing_EMATrend"; Port=8081 },
    @{ Id="02"; Name="BTCSwing_SuperTrend"; Port=8082 },
    @{ Id="03"; Name="BTCSwing_MACDDivergence"; Port=8083 },
    @{ Id="04"; Name="BTCSwing_SqueezeBreakout"; Port=8084 },
    @{ Id="05"; Name="BTCSwing_DonchianTurtle"; Port=8085 },
    @{ Id="06"; Name="BTCSwing_IchimokuCloud"; Port=8086 },
    @{ Id="07"; Name="BTCSwing_RSI_VWAP"; Port=8087 },
    @{ Id="08"; Name="BTCSwing_EMARibbon"; Port=8088 },
    @{ Id="09"; Name="BTCSwing_StochRSI_CMF"; Port=8089 },
    @{ Id="10"; Name="BTCSwing_FibonacciBands"; Port=8090 }
)

$FreqtradeExe = Join-Path $ProjectRoot ".venv\Scripts\freqtrade.exe"
$LogDir = Join-Path $ProjectRoot "user_data\logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

function Test-PortFast ($port) {
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $iar = $tcp.BeginConnect("127.0.0.1", $port, $null, $null)
        $wait = $iar.AsyncWaitHandle.WaitOne(150, $false)
        if (-not $wait) {
            $tcp.Close()
            return $false
        }
        $tcp.EndConnect($iar)
        $tcp.Close()
        return $true
    } catch {
        return $false
    }
}

function Start-Fleet {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "  Starting 10 BTC Swing Trading Bots (Dry-Run, $100 Each)   " -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    foreach ($item in $Strategies) {
        $id = $item.Id
        $name = $item.Name
        $port = $item.Port
        $configFile = "user_data/configs/config_btc_$id.json"
        $logFile = Join-Path $LogDir "bot_$id.log"
        $errFile = Join-Path $LogDir "bot_${id}_err.log"

        if (Test-PortFast $port) {
            Write-Host "[-] Bot $($id) ($name) is ALREADY running on port $port" -ForegroundColor Yellow
            continue
        }

        Write-Host "[+] Launching Bot $($id) ($name) on port $port..." -ForegroundColor Green
        $procArgs = "trade --config $configFile --strategy $name"
        Start-Process -FilePath $FreqtradeExe -ArgumentList $procArgs -WorkingDirectory $ProjectRoot -RedirectStandardOutput $logFile -RedirectStandardError $errFile -WindowStyle Hidden
        Start-Sleep -Milliseconds 400
    }

    Write-Host "`nWaiting for all bots to initialize API listeners..." -ForegroundColor Cyan
    Start-Sleep -Seconds 4
    Get-FleetStatus
}

function Stop-Fleet {
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host "  Stopping all running Freqtrade bot processes...          " -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Yellow

    $processes = Get-Process -Name "*freqtrade*" -ErrorAction SilentlyContinue
    if ($processes) {
        foreach ($p in $processes) {
            Write-Host "[x] Stopping process ID $($p.Id)..." -ForegroundColor Red
            Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        }
    } else {
        Write-Host "No active Freqtrade processes found." -ForegroundColor DarkGray
    }
}

function Get-FleetStatus {
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host "               10-BOT FLEET LIVE STATUS                     " -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    $table = @()
    foreach ($item in $Strategies) {
        $id = $item.Id
        $name = $item.Name
        $port = $item.Port

        $isListening = Test-PortFast $port
        $status = if ($isListening) { "RUNNING" } else { "OFFLINE" }
        $dbPath = "user_data/db/btc_$id.sqlite"
        $dbStatus = if (Test-Path $dbPath) { "Ready" } else { "Pending" }

        $table += [PSCustomObject]@{
            BotId = $id
            Strategy = $name
            Port = $port
            Status = $status
            Database = $dbStatus
        }
    }

    $table | Format-Table -AutoSize
}

switch ($Action) {
    "start" { Start-Fleet }
    "stop"  { Stop-Fleet }
    "restart" { Stop-Fleet; Start-Sleep -Seconds 2; Start-Fleet }
    "status" { Get-FleetStatus }
}
