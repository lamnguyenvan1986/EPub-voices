#!/usr/bin/env python3
"""Đóng gói giọng Piper tiếng Việt của cộng đồng thành gói sherpa-onnx.

Mỗi gói vits-piper-<id>.tar.bz2 có cùng cấu trúc với gói của sherpa-onnx để app
tải và giải nén như giọng VAIS1000:
  vits-piper-<id>/<id>.onnx       model gốc, thêm metadata sherpa-onnx (không đổi trọng số)
  vits-piper-<id>/<id>.onnx.json  cấu hình Piper gốc
  vits-piper-<id>/tokens.txt      tạo từ phoneme_id_map
  vits-piper-<id>/espeak-ng-data  lấy từ gói vits-piper-vi_VN-vais1000-medium
  vits-piper-<id>/MODEL_CARD      nguồn và thay đổi

Chạy: python3 package.py <thư mục ra>   (cần: pip install onnx)
Workflow .github/workflows/offline_voices.yml chạy script này, đưa gói lên bản
phát hành tts-voices và model gốc (cho bản web) lên nhánh web của repo công khai
lamnguyenvan1986/EPub-voices. Danh sách phải khớp offlineViVoices trong
lib/services/neural_tts/offline_voices.dart của repo app (lamnguyenvan1986/EPub).
"""
import hashlib
import json
import os
import shutil
import sys
import tarfile
import urllib.request

import onnx

ESPEAK_SOURCE = ('https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/'
                 'vits-piper-vi_VN-vais1000-medium.tar.bz2')

VOICES = [
    {
        'id': 'vi_VN-infore-medium',
        'model': 'https://raw.githubusercontent.com/phatjkk/vits-tts-vietnamese/'
                 'f6a9647961a030fb0e2c1217a37d9c9daa2ea2ad/fine-tuning-model/v2/finetuning_pretrained_vi.onnx',
        'md5': '2f2ba7c3f36c9142426786bc725adbf1',
        'card': 'Giọng Piper tiếng Việt "InfoRe" (giọng nữ).\n'
                'Nguồn: https://github.com/phatjkk/vits-tts-vietnamese (fine-tuning-model/v2), '
                'tác giả Nguyen Thanh Phat. Model VITS huấn luyện tiếp từ VAIS1000 trên dữ liệu InfoRe.\n'
                'Kho gốc không ghi giấy phép.\n',
    },
    {
        'id': 'vi_VN-misuka-medium',
        'model': 'https://raw.githubusercontent.com/lenhuthuc/Misuka/'
                 '143ff7d8be71d29b482e0b3f147b3d9fd541b929/assets/models/voices/vi_VN-misuka-medium.onnx',
        'md5': '1023f421c28c94bbafb2cad59fdea287',
        'card': 'Giọng Piper tiếng Việt "Misuka" (giọng nữ trầm).\n'
                'Nguồn: https://github.com/lenhuthuc/Misuka (assets/models/voices), Piper 1.5.\n'
                'Kho gốc không ghi giấy phép và không nói rõ dữ liệu huấn luyện.\n',
    },
]

# Giọng Piper tiếng Việt trong thư mục Google Drive của Lam (bản sao các model của
# NGHI-TTS, https://github.com/nghimestudio/nghitts), file chia sẻ "bất kỳ ai có link".
# Đã chấm ngày 2026-11-12 (xem docs/ARCHITECTURE.md). Không lấy: các giọng nhái người
# nổi tiếng có thật (Mỹ Tâm, Trấn Thành, Ngọc Ngân, Việt Thảo, Trump, Ngọc Huyền review
# phim), giọng tiếng Anh/Indonesia (john, mattheo, indo_goreng), kinhphat (đọc sai nhiều),
# calmwoman3688 và taian2 (trùng giọng deepman3909, taian4).
# tên: (id file .onnx, id file .onnx.json, md5 của .onnx)
DRIVE_FILES = {
    'adam1': ('1dSNqezQ358-0K73sfCHLtkc92iXa6dJ2', '15HXFjJy-rxBPbLzsvCkdy2jgYQ0nZ8qm', '7f81762741c4707dbba624addf42f5bc'),
    'banmai': ('1kv_0ab54Mpn2qemXa9so_9Xb9COqH9PB', '1v_N_RFfbY9p1u0inYK5hpVOhQcao1B6x', '6db10e7cc7bc371f2b6d4014115e6707'),
    'chieuthanh': ('11uRsDnfGXfJ3pYsPykev9oFXNvSWL-F7', '16mcUxeS38Lt5pIcGwya2wcsmMJ8nZOv-', '6352c57dfc95f1b15d14c99e44e961ad'),
    'deepman3909': ('1iPai3a4JrQmcK-r9M1n9o5cMH2ty2DqO', '11FBv4Lvu_LTIIUNWGQ2q8xh11erFCaOT', '6837bf93cf15c41ee15bcb4c1579c4a4'),
    'duyoryx3175': ('166WPWIyd13JcxZSue2HAwPRfnK5Mtg_9', '1nn0KvF1T4VCZa4_sTKkDhO7i1dq4J6af', '759ad3cb6421e12b1bb4d94164efe04f'),
    'lacphi': ('16xYL3HAQKSvt5mOVF62mrwUO9lnNZlF2', '1LrY1uMmwQy1a63J5ljTwmAcBA5DwsTiM', 'ef0fb1b7f8ce8c2ab4adc2a95268aa1f'),
    'maiphuong': ('1Nx7eQxl1sW0hWG5EBOaEJWDpJXmHgHPQ', '1qQGNsQFGY72VByat86yvsqX84zn-RQY-', '20cfa2ce763ed8a69d72bde904af5c46'),
    'manhdung': ('1ZLtTmOSmr23bUp57y8Aw3U8xHUV1Zhh2', '1UZ8ZqcEMA-rixO6JtTNjP6mAUlohxCEl', '25e0e0ace2f26189133c7188a62d8f9b'),
    'minhkhang': ('1PeA07oQQ8_-g_BcKo94eD7WsHZoqv8fY', '1vKZTZns9VYbV_rcXQBpqJppMiE7hDOIA', '6c45fae83c566a51ee2f92792764bd83'),
    'minhquang': ('15PohvGDcbOT4mpaP_1cSskfgDeOWLk8r', '1Yg3TAw1Y606dH6PBlheOHmdBF2sG4Mhw', '1a6655a3fbf0c013d6b7a0d1a822a707'),
    'minhthu': ('16eEiRn_WeX6rQDQGe3FJo9l7WqzAd4t-', '1PDZjdx07wUmJHmztfDGs0_acuQeiec7_', '52bf8d1675d3c39a4890a92c9fbbb51d'),
    'phuongtrang': ('1dglLvPgZo4L0GG3HUp5gTf7HJievmIw3', '1Wz_3tMV7OHhLQLKwsReLi4oQ29yOvqfm', '8f871de5a9f3ab53d3c16231c584b01e'),
    'taian4': ('1ToUv49h48KB4JCKyS7Pfc1-Ik1fLDxqM', '1ALd9rmuNsDMmzdbRXxzR-OY8LIcY9mOc', '4df75221cf862d4253cc444e114bf77c'),
    'thanhphuong2': ('1ZiemDlXlM7l2oRkThAAJMwhCF4IUFMXV', '1VDbmdS7UZAKljGQXrOBIx4v7t3L7N-7h', 'bc1842a964beb242175261930ddf2499'),
    'thientam': ('1-YnM54h3DljRBaUgZ7tB3MQgAQhIeIMd', '1Svsj6O9iqDxs2i4sQEyqry_d6guRk63s', '826eecd5e554e409c60a9c5ca8f9fe39'),
    'vunga': ('1LUQVdmslFcxJXxRfa00FcG-wfVUuf6Lu', '13ypn7szTPMumqzDTfJE282_q7IgwfWjP', '9075322a706a41e757c2295ee7428036'),
    'yannew': ('172X73-S1BneYToYB8Eo4gt6k-6buJrcF', '1QtdWtxbn8sj2bmyvZLqP8ZygSfbRErNa', '882be756a38ffc404470a109a3bc9b9f'),
}
DRIVE_CARD = ('Giọng Piper tiếng Việt "{name}" của NGHI-TTS (https://github.com/nghimestudio/nghitts, '
              'mã nguồn Apache-2.0), lấy từ bản sao trên Google Drive của chủ app.\n')
VOICES += [
    {'id': f'vi_VN-{name}-medium', 'drive_model': model, 'drive_config': config, 'md5': digest,
     'card': DRIVE_CARD.format(name=name)}
    for name, (model, config, digest) in DRIVE_FILES.items()
]

CARD_FOOTER = ('\nĐóng gói lại cho sherpa-onnx để dùng trong app LibriAudio: thêm metadata '
               'sherpa-onnx vào .onnx, tạo tokens.txt từ phoneme_id_map, kèm espeak-ng-data của gói '
               'vits-piper-vi_VN-vais1000-medium. Không đổi trọng số.\n')


def drive_url(file_id):
    """Link tải trực tiếp file Google Drive chia sẻ "bất kỳ ai có link"."""
    return f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t'


def fetch(url, dest):
    print('Tải', url, flush=True)
    with urllib.request.urlopen(url) as res, open(dest, 'wb') as f:
        shutil.copyfileobj(res, f)


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def tokens(config):
    """tokens.txt cho sherpa-onnx: mỗi dòng "<ký hiệu> <id>".

    sherpa-onnx chỉ nhận ký hiệu một ký tự; Piper 1.5 thêm vài nguyên âm đôi tiếng Anh
    nhiều ký tự (aɪ, oʊ...) mà espeak tiếng Việt không sinh ra, nên bỏ đi.
    """
    lines = []
    for symbol, ids in config['phoneme_id_map'].items():
        if len(symbol) == 1:
            lines.append(f'{symbol} {ids[0]}\n')
    return ''.join(lines)


def package(voice, espeak_dir, out_dir, work):
    vid = voice['id']
    name = f'vits-piper-{vid}'
    dest = os.path.join(work, name)
    os.makedirs(dest)
    raw = os.path.join(work, f'{vid}.raw.onnx')
    config_path = os.path.join(dest, f'{vid}.onnx.json')
    if 'drive_model' in voice:
        fetch(drive_url(voice['drive_model']), raw)
        fetch(drive_url(voice['drive_config']), config_path)
    else:
        fetch(voice['model'], raw)
        fetch(voice['model'] + '.json', config_path)
    digest = md5(raw)
    print(f'{vid}: md5 {digest}', flush=True)
    if voice.get('md5') and digest != voice['md5']:
        raise SystemExit(f'{vid}: md5 không khớp, model gốc đã đổi')
    with open(config_path, encoding='utf-8') as f:
        config = json.load(f)
    if not config['espeak']['voice'].startswith('vi') or '1' not in config['phoneme_id_map']:
        raise SystemExit(f'{vid}: không phải model tiếng Việt có ký hiệu thanh điệu')

    model = onnx.load(raw)
    meta = {
        'model_type': 'vits', 'comment': 'piper', 'language': 'Vietnamese',
        # vi, vi-vn-x-central (giọng Trung) hoặc vi-vn-x-south (giọng Nam)
        'voice': config['espeak']['voice'],
        'has_espeak': '1', 'n_speakers': str(config.get('num_speakers', 1)),
        'sample_rate': str(config['audio']['sample_rate']),
    }
    del model.metadata_props[:]
    for key, value in meta.items():
        prop = model.metadata_props.add()
        prop.key, prop.value = key, value
    onnx.save(model, os.path.join(dest, f'{vid}.onnx'))

    with open(os.path.join(dest, 'tokens.txt'), 'w', encoding='utf-8') as f:
        f.write(tokens(config))
    with open(os.path.join(dest, 'MODEL_CARD'), 'w', encoding='utf-8') as f:
        f.write(voice['card'] + CARD_FOOTER)
    shutil.copytree(espeak_dir, os.path.join(dest, 'espeak-ng-data'))

    # Bản web: model gốc và cấu hình, workflow đưa lên nhánh web
    # (raw.githubusercontent.com cho phép CORS)
    web_dir = os.path.join(out_dir, 'web')
    os.makedirs(web_dir, exist_ok=True)
    shutil.copy(raw, os.path.join(web_dir, f'{vid}.onnx'))
    shutil.copy(config_path, os.path.join(web_dir, f'{vid}.onnx.json'))

    archive = os.path.join(out_dir, f'{name}.tar.bz2')
    with tarfile.open(archive, 'w:bz2') as tar:
        tar.add(dest, arcname=name)
    print('Xong', archive, flush=True)


def main():
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'build/offline_voices')
    os.makedirs(out_dir, exist_ok=True)
    work = os.path.join(out_dir, 'work')
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    espeak_archive = os.path.join(work, 'vais1000.tar.bz2')
    fetch(ESPEAK_SOURCE, espeak_archive)
    with tarfile.open(espeak_archive) as tar:
        members = [m for m in tar.getmembers() if '/espeak-ng-data' in m.name]
        tar.extractall(work, members=members)
    espeak_dir = os.path.join(work, 'vits-piper-vi_VN-vais1000-medium', 'espeak-ng-data')
    for voice in VOICES:
        package(voice, espeak_dir, out_dir, work)
    shutil.rmtree(work)


if __name__ == '__main__':
    main()
