import sys
import subprocess
import os

if len(sys.argv) < 2:
    print("❗ Использование: python convert_to_wav.py <имя_или_путь_к_файлу>")
    sys.exit(1)

filename = sys.argv[1]

# если указано только имя — ищем в data/raw/lectures/
if not os.path.exists(filename):
    # Попробовать путь относительно каталога скрипта (scripts/../data/raw/lectures/...)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_path_script = os.path.normpath(os.path.join(script_dir, "..", "data", "raw", "lectures", filename))
    # Также попробовать путь относительно текущей рабочей директории (./data/raw/lectures/...)
    possible_path_cwd = os.path.normpath(os.path.join(os.getcwd(), "data", "raw", "lectures", filename))

    if os.path.exists(possible_path_script):
        filename = possible_path_script
    elif os.path.exists(possible_path_cwd):
        filename = possible_path_cwd
    else:
        print(f"❌ Файл не найден: {filename}")
        print("Проверял пути:")
        print(f" - как введён (относительно cwd): {os.path.abspath(filename)}")
        print(f" - относительно скрипта: {possible_path_script}")
        print(f" - относительно cwd/data/raw/lectures: {possible_path_cwd}")
        print("Решения: передайте полный путь к .mp4 или поместите файл в 'data/raw/lectures/'.")
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
