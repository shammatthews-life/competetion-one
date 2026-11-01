$ErrorActionPreference = 'Stop'
$root = 'D:\gci\_gci_push_tmp\champion\EXP85_SUBMISSION_CORRELATION_REPLICATION'
$runner = Join-Path $root 'source\correlation_runner.py'
$reference = Join-Path $root 'reference_artifacts\historical_champion.csv'
$sourcePaths = @(
    @{
        Run = 'id_aligned_v5'
        Candidate = 'candidate_artifacts\track1_v5_rebuilt.csv'
        Original = 'D:\gci\champion\08_SUBMISSIONS\submission_exp85_recovery_v5_rebuilt.csv'
        Note = 'Hybrid historical-cache replay; not a full raw-only rebuild.'
    },
    @{
        Run = 'phase3'
        Candidate = 'candidate_artifacts\track1_phase3_rebuilt.csv'
        Original = 'D:\gci\gemini\champion\phase3_reconstruction\independent_rebuild\submission_exp85_rebuilt.csv'
        Note = 'Phase 3 synthesis uses historical suite caches; mixed lineage.'
    },
    @{
        Run = 'full_rebuilt'
        Candidate = 'candidate_artifacts\track1_full_rebuilt.csv'
        Original = 'D:\gci\champion\08_SUBMISSIONS\submission_exp85_full_rebuilt.csv'
        Note = 'Archived candidate; producer lineage not recertified by correlation.'
    },
    @{
        Run = 'submission_final'
        Candidate = 'candidate_artifacts\submission_final.csv'
        Original = 'D:\gci\gemini\final_submission\submission_final.csv'
        Note = 'Submission origin not established by this comparison.'
    },
    @{
        Run = 'current_track1b'
        Candidate = 'candidate_artifacts\track1b_current_production.csv'
        Original = 'D:\gci\champion\EXP85_FINAL_RECONSTRUCTION\runs\track1b_1793445445960\submission_track1b.csv'
        Note = 'Current production raw-input candidate; not equivalent to historical components.'
    }
)

foreach ($item in $sourcePaths) {
    $runStamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
    $runId = "${runStamp}_$($item.Run)"
    $runDir = Join-Path $root "runs\$runId"
    if (Test-Path -LiteralPath $runDir) {
        $runId = "$runId`_$([guid]::NewGuid().ToString('N'))"
        $runDir = Join-Path $root "runs\$runId"
    }
    $candidate = Join-Path $root $item.Candidate
    & python -B $runner `
        --reference $reference `
        --candidate $candidate `
        --run-dir $runDir `
        --method id_aligned `
        --reference-source-path 'D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv' `
        --candidate-source-path $item.Original `
        --lineage-note $item.Note
    if ($LASTEXITCODE -ne 0) {
        throw "Correlation run failed: $runId"
    }
}
