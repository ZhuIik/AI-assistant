param(
    [string]$SegmentsDir = (Join-Path $PSScriptRoot "..\data\segments"),
    [string]$AnnotationsDir = (Join-Path $PSScriptRoot "..\data\annotations"),
    [string[]]$LectureId = @(),
    [switch]$SkipExisting,
    [int]$MaxKeywords = 5,
    [double]$MaxGapSec = 12.0,
    [double]$MaxBlockSec = 180.0,
    [int]$MaxChars = 2200,
    [double]$ShortTurnMaxSec = 4.5,
    [int]$ShortTurnMaxTokens = 8,
    [double]$MergeShortTurnGapSec = 6.0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$null = New-Item -ItemType Directory -Force -Path $AnnotationsDir

$shortTurnMarkers = @(
    "да", "ага", "угу", "окей", "хорошо", "ладно", "понятно", "ясно",
    "спасибо", "так", "вот", "ну", "нет", "конечно", "отлично"
)

$stopwords = @(
    "а", "без", "более", "бы", "был", "была", "были", "было", "быть", "в", "вам", "вас",
    "весь", "во", "вот", "все", "всё", "вы", "где", "да", "даже", "для", "до", "его", "ее",
    "её", "если", "есть", "еще", "ещё", "же", "за", "здесь", "и", "из", "или", "им", "их",
    "к", "как", "какая", "какие", "какой", "когда", "кто", "ли", "либо", "мне", "может",
    "можно", "мы", "на", "над", "надо", "нам", "нас", "не", "него", "нее", "неё", "нет",
    "ни", "но", "ну", "о", "об", "однако", "он", "она", "они", "оно", "опять", "от", "по",
    "под", "при", "про", "просто", "раз", "с", "сам", "сама", "сами", "само", "себя", "сейчас",
    "сказал", "сказать", "со", "совсем", "так", "также", "такой", "там", "тебя", "тем", "то",
    "тогда", "того", "тоже", "только", "том", "тот", "тут", "ты", "у", "уже", "хорошо", "чего",
    "чей", "чем", "что", "чтобы", "эта", "эти", "это", "я",
    "типа", "вроде", "короче", "вообще", "прям", "самый", "самое", "самая",
    "этот", "сюда", "туда", "этого", "этом", "этой", "грубо", "говоря", "ладно", "угу", "ага",
    "окей", "давайте", "пожалуйста"
)

function Normalize-Text {
    param([AllowNull()][string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) {
        return ""
    }
    return ([regex]::Replace($Text, "\s+", " ")).Trim()
}

function Tokenize {
    param([AllowNull()][string]$Text)
    $matches = [regex]::Matches(([string]$Text).ToLowerInvariant(), "[\p{L}\p{Nd}-]+")
    return @($matches | ForEach-Object { $_.Value })
}

function Looks-LikeShortTurn {
    param(
        [string]$Text,
        [double]$DurationSec,
        [double]$MaxSec,
        [int]$MaxTokens
    )

    $tokens = @(Tokenize (Normalize-Text $Text))
    if ($tokens.Count -eq 0) {
        return $true
    }
    if ($DurationSec -le $MaxSec -and $tokens.Count -le $MaxTokens) {
        return $true
    }
    if ($tokens.Count -le 3) {
        $unknown = @($tokens | Where-Object { $shortTurnMarkers -notcontains $_ })
        if ($unknown.Count -eq 0) {
            return $true
        }
    }
    return $false
}

function New-EmptyBlock {
    param([string]$LectureName)
    return @{
        lecture_id     = $LectureName
        segment_ids    = @()
        start          = $null
        end            = $null
        start_sec      = $null
        end_sec        = $null
        speaker_raw    = $null
        speakers_raw   = @()
        speaker_counts = @{}
        text           = ""
    }
}

function Get-DominantSpeaker {
    param(
        [hashtable]$SpeakerCounts,
        [object[]]$SpeakersRaw
    )

    $bestSpeaker = $null
    $bestCount = -1
    $bestIndex = [int]::MaxValue

    for ($i = 0; $i -lt $SpeakersRaw.Count; $i++) {
        $speaker = [string]$SpeakersRaw[$i]
        $count = [int]$SpeakerCounts[$speaker]
        if ($count -gt $bestCount -or ($count -eq $bestCount -and $i -lt $bestIndex)) {
            $bestSpeaker = $speaker
            $bestCount = $count
            $bestIndex = $i
        }
    }

    return $bestSpeaker
}

function Register-Speaker {
    param(
        [hashtable]$Block,
        [AllowNull()][string]$Speaker
    )

    if ([string]::IsNullOrWhiteSpace($Speaker)) {
        return
    }
    if (-not $Block.speaker_counts.ContainsKey($Speaker)) {
        $Block.speaker_counts[$Speaker] = 0
        $Block.speakers_raw += $Speaker
    }
    $Block.speaker_counts[$Speaker] = [int]$Block.speaker_counts[$Speaker] + 1
    $Block.speaker_raw = Get-DominantSpeaker -SpeakerCounts $Block.speaker_counts -SpeakersRaw $Block.speakers_raw
}

function Append-Segment {
    param(
        [hashtable]$Block,
        [pscustomobject]$Segment,
        [string]$Text
    )

    if ($Block.segment_ids.Count -eq 0) {
        $Block.start = $Segment.start
        $Block.start_sec = [double]$Segment.start_sec
    }

    $Block.segment_ids += [int]$Segment.segment_id
    $Block.end = $Segment.end
    $Block.end_sec = [double]$Segment.end_sec
    $Block.text = if ([string]::IsNullOrWhiteSpace($Block.text)) { $Text } else { "$($Block.text) $Text" }
    Register-Speaker -Block $Block -Speaker $Segment.speaker
}

function Should-Split {
    param(
        [hashtable]$Current,
        [pscustomobject]$Segment
    )

    if ($Current.segment_ids.Count -eq 0) {
        return $false
    }

    $sameSpeaker = $Current.speaker_raw -eq $Segment.speaker
    $gap = [double]$Segment.start_sec - [double]$Current.end_sec
    $newDuration = [double]$Segment.end_sec - [double]$Current.start_sec
    $segmentText = Normalize-Text $Segment.text
    $newChars = $Current.text.Length + $(if ($Current.text.Length -gt 0) { 1 } else { 0 }) + $segmentText.Length
    $shortTurn = Looks-LikeShortTurn -Text $segmentText -DurationSec ([double]$Segment.duration_sec) -MaxSec $ShortTurnMaxSec -MaxTokens $ShortTurnMaxTokens

    if ($gap -gt $MaxGapSec -or $newDuration -gt $MaxBlockSec -or $newChars -gt $MaxChars) {
        return $true
    }

    if (-not $sameSpeaker -and -not $shortTurn) {
        return $true
    }

    return $false
}

function Merge-Blocks {
    param(
        [hashtable]$Left,
        [hashtable]$Right
    )

    $merged = New-EmptyBlock -LectureName $Left.lecture_id
    $merged.start = $Left.start
    $merged.start_sec = $Left.start_sec
    $merged.end = $Right.end
    $merged.end_sec = $Right.end_sec
    $merged.segment_ids = @($Left.segment_ids + $Right.segment_ids)
    $merged.text = Normalize-Text "$($Left.text) $($Right.text)"

    foreach ($speaker in $Left.speakers_raw) {
        for ($i = 0; $i -lt [int]$Left.speaker_counts[$speaker]; $i++) {
            Register-Speaker -Block $merged -Speaker $speaker
        }
    }
    foreach ($speaker in $Right.speakers_raw) {
        for ($i = 0; $i -lt [int]$Right.speaker_counts[$speaker]; $i++) {
            Register-Speaker -Block $merged -Speaker $speaker
        }
    }

    return $merged
}

function Merge-ShortBlocks {
    param([object[]]$RawBlocks)

    $merged = New-Object System.Collections.Generic.List[object]
    foreach ($block in $RawBlocks) {
        if ($merged.Count -eq 0) {
            $merged.Add($block)
            continue
        }

        $previous = [hashtable]$merged[$merged.Count - 1]
        $gap = [double]$block.start_sec - [double]$previous.end_sec
        $duration = [double]$block.end_sec - [double]$block.start_sec
        $isShortBlock = Looks-LikeShortTurn -Text $block.text -DurationSec $duration -MaxSec $ShortTurnMaxSec -MaxTokens $ShortTurnMaxTokens

        if ($gap -le $MergeShortTurnGapSec -and $isShortBlock) {
            $merged[$merged.Count - 1] = Merge-Blocks -Left $previous -Right $block
        }
        else {
            $merged.Add($block)
        }
    }

    return @($merged.ToArray())
}

function Get-ActorLabel {
    param([pscustomobject]$Item)

    if ($Item.speaker -eq "professor") {
        return "Преподаватель"
    }
    if ($Item.speaker -eq "student") {
        return "Студент"
    }
    if (@($Item.speakers_raw).Count -gt 1) {
        return "Участники обсуждения"
    }
    return "В этом фрагменте"
}

function Extract-Keywords {
    param(
        [string]$Text,
        [int]$Limit
    )

    $counts = @{}
    foreach ($token in (Tokenize $Text)) {
        if ($token.Length -lt 5) {
            continue
        }
        if ($stopwords -contains $token) {
            continue
        }
        if ($token -match '^\d+$') {
            continue
        }
        $key = $token.ToLowerInvariant()
        if (-not $counts.ContainsKey($key)) {
            $counts[$key] = 0
        }
        $counts[$key] = [int]$counts[$key] + 1
    }

    return @(
        $counts.GetEnumerator() |
        Sort-Object @{ Expression = { $_.Value }; Descending = $true }, @{ Expression = { $_.Key }; Descending = $false } |
        Select-Object -First $Limit |
        ForEach-Object { $_.Key }
    )
}

function Looks-LikeRealQuestion {
    param([string]$Text)
    $tokens = @(Tokenize $Text)
    $meaningful = @($tokens | Where-Object { $_.Length -ge 4 -and $stopwords -notcontains $_ })
    return $Text.Length -ge 6 -and $Text.Length -le 200 -and $tokens.Count -gt 1 -and $meaningful.Count -ge 1
}

function Extract-Questions {
    param([string]$Text)

    $found = New-Object System.Collections.Generic.List[string]
    foreach ($part in [regex]::Split($Text, '(?<=[\?\!\.])\s+')) {
        $clean = Normalize-Text $part
        if ($clean.Contains("?") -and (Looks-LikeRealQuestion $clean)) {
            $found.Add($clean)
        }
    }

    $seen = @{}
    $unique = New-Object System.Collections.Generic.List[string]
    foreach ($question in $found) {
        if (-not $seen.ContainsKey($question)) {
            $seen[$question] = $true
            $unique.Add($question)
        }
        if ($unique.Count -ge 5) {
            break
        }
    }

    return @($unique)
}

function Infer-InteractionType {
    param([string]$Text)
    $lowered = $Text.ToLowerInvariant()
    if ($lowered.Contains("например")) {
        return "example"
    }
    if ($lowered -match 'сделайте|посчитайте|откройте|задание|упражнен') {
        return "exercise"
    }
    if ($Text.Contains("?")) {
        return "qa"
    }
    return "lecture"
}

function Infer-CognitiveLevel {
    param(
        [string]$Text,
        [string]$InteractionType
    )
    $lowered = $Text.ToLowerInvariant()

    if ($lowered -match 'разработ|сформулир|построй|созда') {
        return "create"
    }
    if ($lowered -match 'оцен|выберите|сравнит|лучше|хуже') {
        return "evaluate"
    }
    if ($lowered -match 'анализ|разбер|структур|связ|причин') {
        return "analyze"
    }
    if ($InteractionType -eq "exercise" -or $lowered -match 'примен|рассчита|использу') {
        return "apply"
    }
    if ($lowered -match 'почему|объясн|суть|значит|понят') {
        return "understand"
    }
    if ($InteractionType -eq "qa") {
        return "remember"
    }
    return "understand"
}

function Infer-Difficulty {
    param(
        [string]$Text,
        [string[]]$Keywords,
        [string]$CognitiveLevel
    )

    $tokenCount = @(Tokenize $Text).Count
    $keywordFactor = [math]::Min($Keywords.Count / 5.0, 1.0)
    $lengthFactor = [math]::Min($tokenCount / 120.0, 1.0)
    $cognitiveFactor = switch ($CognitiveLevel) {
        "remember" { 0.20 }
        "understand" { 0.35 }
        "apply" { 0.55 }
        "analyze" { 0.70 }
        "evaluate" { 0.82 }
        "create" { 0.90 }
        default { 0.40 }
    }

    $score = 0.45 * $cognitiveFactor + 0.30 * $keywordFactor + 0.25 * $lengthFactor
    return [math]::Round([math]::Min([math]::Max($score, 0.05), 0.95), 2)
}

function Infer-Summary {
    param(
        [pscustomobject]$Item,
        [string]$Topic,
        [string[]]$Keywords
    )

    $actor = Get-ActorLabel -Item $Item
    if ($Topic -eq "Обсуждение учебного материала" -and $Keywords.Count -gt 0) {
        $sample = ($Keywords | Select-Object -First 3) -join ", "
        return "$actor обсуждают учебный материал, связанный с: $sample."
    }
    if ($Topic -eq "Фрагмент лекции") {
        return "$actor обсуждают короткий фрагмент лекции без явно выраженной отдельной темы."
    }
    return "$actor обсуждают тему «$Topic»."
}

function Build-BlockAnnotations {
    param([hashtable]$Block)

    $text = Normalize-Text $Block.text
    $keywords = @(Extract-Keywords -Text $text -Limit $MaxKeywords)
    $topic = if ($keywords.Count -gt 0) { "Обсуждение учебного материала" } else { "Фрагмент лекции" }
    $interactionType = Infer-InteractionType -Text $text
    $cognitiveLevel = Infer-CognitiveLevel -Text $text -InteractionType $interactionType
    $questions = @(Extract-Questions -Text $text)
    $speaker = $null

    $item = [pscustomobject][ordered]@{
        id              = 0
        lecture_id      = $Block.lecture_id
        segment_ids     = @($Block.segment_ids)
        topic           = $topic
        keywords        = @($keywords)
        questions       = @($questions)
        start           = $Block.start
        end             = $Block.end
        summary         = ""
        speaker         = $speaker
        speaker_raw     = $Block.speaker_raw
        speakers_raw    = @($Block.speakers_raw)
        cognitive_level = $cognitiveLevel
        interaction_type = $interactionType
        embedding       = @()
        difficulty      = 0.0
        text            = $text
    }

    $item.summary = Infer-Summary -Item $item -Topic $topic -Keywords $keywords
    $item.difficulty = Infer-Difficulty -Text $text -Keywords $keywords -CognitiveLevel $cognitiveLevel
    return $item
}

function Convert-SegmentsToBlocks {
    param([pscustomobject[]]$Segments)

    if ($Segments.Count -eq 0) {
        return @()
    }

    $lectureName = [string]$Segments[0].lecture_id
    $rawBlocks = New-Object System.Collections.Generic.List[object]
    $current = New-EmptyBlock -LectureName $lectureName

    foreach ($segment in $Segments) {
        $text = Normalize-Text $segment.text
        if (Should-Split -Current $current -Segment $segment) {
            $rawBlocks.Add($current)
            $current = New-EmptyBlock -LectureName $lectureName
        }
        Append-Segment -Block $current -Segment $segment -Text $text
    }

    if ($current.segment_ids.Count -gt 0) {
        $rawBlocks.Add($current)
    }

    return @($rawBlocks.ToArray())
}

function Write-JsonLines {
    param(
        [string]$Path,
        [object[]]$Items
    )

    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    $lines = New-Object System.Collections.Generic.List[string]
    foreach ($item in $Items) {
        $lines.Add(($item | ConvertTo-Json -Compress -Depth 8))
    }
    [System.IO.File]::WriteAllLines($Path, ([string[]]$lines.ToArray()), $utf8NoBom)
}

$segmentFiles = @(Get-ChildItem -Path $SegmentsDir -Filter "*.segments.jsonl" | Sort-Object Name)
if ($LectureId.Count -gt 0) {
    $allowed = @{}
    foreach ($name in $LectureId) {
        $allowed[$name] = $true
    }
    $segmentFiles = @($segmentFiles | Where-Object { $allowed.ContainsKey(($_.BaseName -replace '\.segments$', '')) })
}

if ($segmentFiles.Count -eq 0) {
    throw "No segment files found in $SegmentsDir"
}

$totalBlocks = 0
$totalAnnotations = 0

foreach ($file in $segmentFiles) {
    $lectureName = $file.BaseName -replace '\.segments$', ''
    $draftPath = Join-Path $AnnotationsDir "$lectureName.draft.jsonl"
    $annotationPath = Join-Path $AnnotationsDir "$lectureName.annotations.jsonl"

    if ($SkipExisting -and (Test-Path $annotationPath)) {
        Write-Host "[skip] $annotationPath already exists"
        continue
    }

    $segments = @(
        Get-Content -Path $file.FullName -Encoding utf8 |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { $_ | ConvertFrom-Json }
    )

    $blocks = @(Convert-SegmentsToBlocks -Segments $segments)
    $draftItems = New-Object System.Collections.Generic.List[object]
    $annotationItems = New-Object System.Collections.Generic.List[object]

    for ($i = 0; $i -lt $blocks.Count; $i++) {
        $block = $blocks[$i]
        $draft = [pscustomobject][ordered]@{
            id              = $i + 1
            lecture_id      = $block.lecture_id
            segment_ids     = @($block.segment_ids)
            topic           = ""
            keywords        = @()
            questions       = @()
            start           = $block.start
            end             = $block.end
            summary         = ""
            speaker         = $null
            speaker_raw     = $block.speaker_raw
            speakers_raw    = @($block.speakers_raw)
            cognitive_level = ""
            interaction_type = ""
            embedding       = @()
            difficulty      = $null
            text            = $block.text
        }
        $annotation = Build-BlockAnnotations -Block $block
        $annotation.id = $i + 1

        $draftItems.Add($draft)
        $annotationItems.Add($annotation)
    }

    Write-JsonLines -Path $draftPath -Items ([object[]]$draftItems.ToArray())
    Write-JsonLines -Path $annotationPath -Items ([object[]]$annotationItems.ToArray())

    $totalBlocks += $draftItems.Count
    $totalAnnotations += $annotationItems.Count

    Write-Host "[done] $draftPath ($($draftItems.Count) blocks)"
    Write-Host "[done] $annotationPath ($($annotationItems.Count) annotations)"
}

Write-Host "Draft annotations built: $totalBlocks"
Write-Host "Prefilled annotations built: $totalAnnotations"
