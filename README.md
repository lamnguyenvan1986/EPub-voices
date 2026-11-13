# EPub-voices

Giọng đọc offline tiếng Việt (Piper, chạy bằng sherpa-onnx) cho app LibriAudio.
Repo này để công khai để app tải giọng không cần đăng nhập; code app nằm ở repo khác.

- `package.py`: tải model gốc (NGHI-TTS, InfoRe, Misuka), kiểm tra md5, đóng gói thành `vits-piper-<id>.tar.bz2`.
- `.github/workflows/offline_voices.yml`: chạy `package.py` khi sửa script hoặc chạy tay, rồi
  - đưa gói lên bản phát hành [`tts-voices`](https://github.com/lamnguyenvan1986/EPub-voices/releases/tag/tts-voices) (app mobile tải ở đây),
  - đưa model gốc `.onnx` + `.onnx.json` lên nhánh `web` (bản web tải qua raw.githubusercontent.com, có CORS).

Nguồn và giấy phép của từng giọng ghi trong file `MODEL_CARD` trong mỗi gói.
Danh sách giọng phải khớp `offlineViVoices` trong `lib/services/neural_tts/offline_voices.dart` của app.
