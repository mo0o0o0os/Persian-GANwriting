"""
بررسی سلامت دیتاست آبان روی یک پوشه‌ی داخل گوگل‌درایو — برای اجرا در Google Colab.

دقیقاً همان بررسی نسخه‌ی محلی (tools/check_dataset_local.py) را انجام می‌دهد،
با این تفاوت که اول گوگل‌درایو را mount می‌کند و پوشه‌ی هدف را از آن‌جا می‌خواند.

نحوه‌ی استفاده در کولب:
  ۱) این فایل را در یک سلول کولب paste کنید (یا آپلودش کنید و !python check_dataset_drive.py بزنید)
  ۲) مقدار IMAGES_DIR را زیر «تنظیمات» با مسیر واقعی پوشه‌ی عکس‌ها در درایوتان جایگزین کنید
  ۳) اجرا کنید

⚠️ چون درایو از طریق FUSE mount می‌شود، خواندن ۶۲ هزار نام فایل ممکن است چند دقیقه طول بکشد —
طبیعی است، صبر کنید.
"""

import os
import re
import csv
import sys
import urllib.request
from collections import Counter

# ============================== تنظیمات ==============================
IMAGES_DIR = "/content/drive/MyDrive/PUT_YOUR_WORDS_FOLDER_PATH_HERE"   # پوشه‌ی عکس‌ها داخل درایو

WORD_CSV = "/content/word_labels_with_translations.csv"     # اگر ندارید، خودکار از گیت‌هاب دانلود می‌شود
GT_FILES = [                                                  # اختیاری؛ اگر ندارید [] بگذارید (خودکار هم دانلود می‌شوند)
    "/content/gan.aban.tr_va.gt.filter27",
    "/content/gan.aban.test.gt.filter27",
]

OUTPUT_DIR = "/content/dataset_check_report"
REPO_RAW_BASE = "https://raw.githubusercontent.com/mo0o0o0os/Persian-GANwriting/claude/friendly-ride-to6rjl"
# =======================================================================

FNAME_RE = re.compile(r"^ID(\d+)_(.+)\.png$", re.IGNORECASE)


def mount_drive_if_needed():
    if "google.colab" not in sys.modules:
        print("⚠️ این اسکریپت برای Google Colab نوشته شده؛ محیط فعلی کولب تشخیص داده نشد — ادامه می‌دهیم "
              "به این امید که IMAGES_DIR از قبل در دسترس باشد.")
        return
    if not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive
        print("در حال mount کردن گوگل‌درایو ...")
        drive.mount("/content/drive")
    else:
        print("گوگل‌درایو از قبل mount شده است.")


def ensure_file(local_path, raw_url):
    if os.path.exists(local_path):
        return local_path
    print(f"⚠️ {local_path} پیدا نشد؛ در حال دانلود از {raw_url} ...")
    try:
        urllib.request.urlretrieve(raw_url, local_path)
        print("   دانلود شد.")
        return local_path
    except Exception as e:
        print(f"   ❌ دانلود ناموفق بود: {e}")
        return None


def main():
    mount_drive_if_needed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ۱) ۱۲۵ کد کلمه را از CSV بخوان
    csv_path = ensure_file(WORD_CSV, f"{REPO_RAW_BASE}/word_labels_with_translations.csv")
    if not csv_path:
        raise SystemExit("فایل CSV کلمات در دسترس نیست؛ مسیر WORD_CSV را دستی تنظیم کنید.")
    word_codes = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            word_codes.append(row["Word ID"].strip())
    word_codes = sorted(set(word_codes))
    print(f"تعداد کلمات در CSV: {len(word_codes)}")

    # ۲) فایل‌های پوشه‌ی درایو را یک‌بار لیست کن (ممکن است چند دقیقه طول بکشد)
    if not os.path.isdir(IMAGES_DIR):
        raise SystemExit(f"پوشه‌ی {IMAGES_DIR} پیدا نشد. مقدار IMAGES_DIR را در بالای اسکریپت درست کنید.")
    print("در حال خواندن لیست فایل‌های پوشه‌ی درایو (ممکن است کند باشد، صبر کنید) ...")
    all_files = os.listdir(IMAGES_DIR)
    print(f"تعداد کل فایل‌ها در پوشه: {len(all_files)}")

    # ۳) پارس کردن نام فایل‌ها
    per_word = Counter()
    per_writer = Counter()
    found_pairs = set()
    bad_names = []

    for fname in all_files:
        m = FNAME_RE.match(fname)
        if not m:
            bad_names.append(fname)
            continue
        wid, word = m.group(1), m.group(2)
        per_word[word] += 1
        per_writer[wid] += 1
        found_pairs.add((wid, word))

    writer_ids_seen = set(per_writer)
    print(f"تعداد نویسندگان یافت‌شده: {len(writer_ids_seen)}")
    print(f"تعداد فایل با نام غیرمنتظره (الگوی ID<شماره>_<کد کلمه>.png را ندارند): {len(bad_names)}")
    if bad_names:
        print("  نمونه:", bad_names[:10])

    # ۴) گزارش هر کلمه
    expected_per_word = len(writer_ids_seen)
    word_mismatch = [(w, per_word.get(w, 0)) for w in word_codes if per_word.get(w, 0) != expected_per_word]
    print(f"\n--- کلماتی با تعداد تصویر غیرمنتظره (انتظار≈{expected_per_word}) ---")
    print(f"{len(word_mismatch)} از {len(word_codes)} کلمه")
    for w, c in sorted(word_mismatch, key=lambda x: x[1])[:30]:
        print(f"  {w}: {c}")

    words_not_in_csv = sorted(set(per_word) - set(word_codes))
    words_missing_entirely = [w for w in word_codes if per_word.get(w, 0) == 0]
    if words_not_in_csv:
        print(f"\n⚠️ کدهایی که در نام فایل‌ها هست ولی در CSV نیست ({len(words_not_in_csv)}): {words_not_in_csv[:10]}")
    if words_missing_entirely:
        print(f"⚠️ کلماتی که اصلاً هیچ تصویری از آن‌ها نیست ({len(words_missing_entirely)}): {words_missing_entirely}")

    # ۵) گزارش هر نویسنده
    writer_mismatch = [(wid, c) for wid, c in per_writer.items() if c != len(word_codes)]
    print(f"\n--- نویسندگانی که تعداد عکسشان دقیقاً {len(word_codes)} نیست ---")
    print(f"{len(writer_mismatch)} از {len(writer_ids_seen)} نویسنده")
    for wid, c in sorted(writer_mismatch, key=lambda x: x[1])[:30]:
        print(f"  {wid}: {c}")

    # ۶) مقایسه با گراندتروث رسمی ریپو
    gt_paths = []
    for i, gtf in enumerate(GT_FILES):
        default_name = "gan.aban.tr_va.gt.filter27" if i == 0 else "gan.aban.test.gt.filter27"
        resolved = ensure_file(gtf, f"{REPO_RAW_BASE}/Groundtruth_farsi/{default_name}")
        if resolved:
            gt_paths.append(resolved)

    if gt_paths:
        expected_idx = set()
        for gtf in gt_paths:
            with open(gtf, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    idpart = line.strip().split(" ", 1)[0]
                    _, idx = idpart.split(",")
                    expected_idx.add(idx)

        found_idx = {f"ID{wid}_{word}" for wid, word in found_pairs}
        missing_from_gt = sorted(expected_idx - found_idx)
        extra_not_in_gt = sorted(found_idx - expected_idx)

        print(f"\n--- مقایسه با گراندتروث رسمی ریپو ({len(expected_idx)} فایلی که کد آموزش واقعاً نیاز دارد) ---")
        print(f"فایل‌های مورد نیاز که در پوشه‌ی درایو شما نیست: {len(missing_from_gt)}")
        if missing_from_gt:
            out_path = os.path.join(OUTPUT_DIR, "missing_from_groundtruth.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(n + ".png" for n in missing_from_gt))
            print("  لیست کامل (نام فایل‌های دقیق):", out_path)
            print("  نمونه:", missing_from_gt[:10])
        print(f"فایل‌های اضافیِ شما که در گراندتروث رسمی نیستند (اشکالی ندارد، فقط اطلاعاتی): {len(extra_not_in_gt)}")
        if not missing_from_gt:
            print("✅ همه‌ی فایل‌های مورد نیاز کد آموزش موجودند — دیتاست برای آموزش کامل است،")
            print("   حتی اگر کل تعداد آن با ۶۲۵۰۰ فرق داشته باشد (چون گراندتروث رسمی خودِ ریپو هم ۶۲۴۸۱ ردیف دارد،")
            print("   یعنی ۱۹ نمونه از ابتدا با فیلتر filter27 کنار گذاشته شده‌اند، نه این‌که شما چیزی کم داشته باشید).")

    # ۷) خروجی کامل برای بررسی دستی
    with open(os.path.join(OUTPUT_DIR, "per_word_counts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["word_code", "count"])
        for word in word_codes:
            w.writerow([word, per_word.get(word, 0)])
    with open(os.path.join(OUTPUT_DIR, "per_writer_counts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["writer_id", "count"])
        for wid in sorted(writer_ids_seen):
            w.writerow([wid, per_writer.get(wid, 0)])

    print(f"\n✅ گزارش کامل (per_word_counts.csv, per_writer_counts.csv, ...) در «{OUTPUT_DIR}» ذخیره شد.")
    print("   برای دانلود گزارش از کولب: از پنل Files سمت چپ روی /content/dataset_check_report راست‌کلیک و دانلود کنید.")


if __name__ == "__main__":
    main()
