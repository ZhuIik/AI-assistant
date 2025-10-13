import sys
import subprocess
import os

if len(sys.argv) < 2:
    print("❗ Использование: python convert_to_wav.py <имя_или_путь_к_файлу>")
    sys.exit(1)

filename = sys.argv[1]

# если указано только имя — ищем в data/raw/
if not os.path.exists(filename):
    possible_path = os.path.join("..", "data", "raw", filename)
    if os.path.exists(possible_path):
        filename = possible_path
    else:
        print(f"❌ Файл не найден: {filename}")
        sys.exit(1)

# создаём имя для .wav рядом с исходником
base, _ = os.path.splitext(filename)
output_file = base + ".wav"

# команда ffmpeg
command = [
    r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
    "-i", filename,
    "-ss", "00:00:00",
    "-ar", "16000",
    "-ac", "1",
    "-c:a", "pcm_s16le",
    output_file
]


# выполняем
subprocess.run(command, check=True)
print(f"✅ Готово! WAV сохранён рядом с видео:\n{os.path.abspath(output_file)}")
