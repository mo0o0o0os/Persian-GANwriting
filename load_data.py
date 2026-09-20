import os
import torch.utils.data as D
import random
import cv2
import numpy as np
from pairs_idx_wid_ABAN import wid2label_tr, wid2label_te
import albumentations as A  # 🟢 کتابخانه جدید را وارد کنید

CREATE_PAIRS = False

IMG_HEIGHT = 64
IMG_WIDTH = 216
MAX_CHARS = 12
NUM_CHANNEL = 15
EXTRA_CHANNEL = NUM_CHANNEL + 1
NUM_WRITERS = 500  # ABAN: 350 train + 150 test
# ABAN splits train/test by DISJOINT writers (unlike IAM, which splits by word
# for the same writers). The writer classifier (cla) is only ever trained on
# wid2label_tr, so its output layer must match len(wid2label_tr) (350), not
# NUM_WRITERS (500) -- otherwise ~150 output classes never receive gradient
# and only add noise to the softmax used for both the real cla_update and the
# generator's adversarial cla loss.
NUM_TRAIN_WRITERS = len(wid2label_tr)
NORMAL = True
OUTPUT_MAX_LEN = MAX_CHARS + 2  # <GO>+groundtruth+<END> = 14

"""مسیر تصاویر کلمات ABAN"""
img_base = "./datasets/aban/words"

# CLOSED_VOCAB=True: هدف مسئله عوض شده از OOV واقعی به
# "style transfer با واژگان بسته" -- یعنی کلمه‌ی هدف (label_xt) هم از
# همون ۱۲۵ کلمه‌ی دیتاست آبان انتخاب می‌شه، نه از یک پیکره‌ی ۴۶هزارتایی
# خارجی. این باعث می‌شه توزیع "واقعی" (۱۲۵ شکل کلمه) و توزیع محتوایی که
# discriminator/recognizer باهاش مواجه می‌شن یکی باشن -- discriminator
# دیگه نمی‌تونه صرفاً با رد کردن "شکل‌های ناآشنا" تقلب کنه، که یکی از
# دلایل اصلی mode collapse روی این دیتاست بود (چون IAM ~۶۰۰۰+ کلمه‌ی
# یکتا داشت ولی آبان فقط ۱۲۵ تا).
# برای برگشت به حالت OOV واقعی (نیازمند یک پیکره‌ی بزرگ‌تر مکمل مثل
# Khayyam)، CLOSED_VOCAB را False کنید.
CLOSED_VOCAB = True
if CLOSED_VOCAB:
    text_corpus_path = "./corpora_farsi/in_vocab.aban.txt"
else:
    text_corpus_path = "./corpora_farsi/corpora-farsi-myChars.tr"

with open(text_corpus_path, "r", encoding='utf-8') as _f:
    text_corpus = _f.read().strip().split('\n')

src = "Groundtruth_farsi/gan.aban.tr_va.gt.filter27"
tar = "Groundtruth_farsi/gan.aban.test.gt.filter27"


def labelDictionary():
    """ساخت دیکشنری الفبای فارسی"""
    ZWNJ = '\u200c'  # Zero-Width Non-Joiner
    labels = list('آابتثجحخدذرزسشصضعفقلمنهوپچکگی' + ZWNJ)
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter


num_classes, letter2index, index2letter = labelDictionary()
tokens = {"GO_TOKEN": 0, "END_TOKEN": 1, "PAD_TOKEN": 2}
num_tokens = len(tokens.keys())
vocab_size = num_classes + num_tokens


def edits1(word, min_len=2, max_len=MAX_CHARS):
    """ایجاد یک ویرایش (edit distance 1) از کلمه فارسی"""
    ZWNJ = '\u200c'
    letters = list('آابتثجحخدذرزسشصضعفقلمنهوپچکگی' + ZWNJ)
    
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    
    visible_len = len([c for c in word if c != ZWNJ])
    
    if visible_len <= min_len:
        candidates = list(set(transposes + replaces + inserts))
    elif visible_len >= max_len:
        candidates = list(set(deletes + transposes + replaces))
    else:
        candidates = list(set(deletes + transposes + replaces + inserts))
    
    if not candidates:
        return word
    return random.choice(candidates)


class ABAN_words(D.Dataset):
    def __init__(self, data_dict, oov, is_train=True):
        self.data_dict = data_dict
        self.oov = oov
        self.output_max_len = OUTPUT_MAX_LEN
        self.is_train = is_train

        # تعریف pipeline برای augmentation
        # نسخه اصلاح شده برای رفع warning ها
        if self.is_train:
            self.transform = A.Compose([
                # 🟢 استفاده از BORDER_REFLECT_101 برای ایجاد بافت طبیعی در لبه‌ها
                A.Rotate(limit=2, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                
                # 🟢 استفاده از BORDER_REFLECT_101 برای تبدیل برشی (shear)
                A.Affine(shear=(-3, 3), interpolation=1,
                         border_mode=cv2.BORDER_REFLECT_101, p=0.5),
                
                # 🟢 استفاده از BORDER_REFLECT_101 برای تبدیل کشسانی
                A.ElasticTransform(alpha=1, sigma=20,
                                   border_mode=cv2.BORDER_REFLECT_101, p=0.3),
                
            ])
        else:
            # در حالت تست، هیچ تبدیلی اعمال نمی‌شود
            self.transform = None

    def __len__(self):
        return len(self.data_dict)

    def new_ed1(self, word_ori):
        word = word_ori.copy()
        # تبدیل numpy array به list
        if isinstance(word, np.ndarray):
            word = word.tolist()
        
        start = word.index(tokens["GO_TOKEN"])
        fin = word.index(tokens["END_TOKEN"])
        word = "".join([index2letter[i - num_tokens] for i in word[start + 1 : fin]])
        new_word = edits1(word)
        label = np.array(self.label_padding(new_word, num_tokens))
        return label
    
    def __getitem__(self, wid_idx_num):
        words = self.data_dict[wid_idx_num]
        np.random.shuffle(words)

        wids = list()
        idxs = list()
        imgs = list()
        img_widths = list()
        labels = list()

        for word in words:
            wid, idx = word[0].split(",")
            img, img_width = self.read_image_single(idx)
            
            # متن کلمه = همه چیز بعد از ID (بدون فاصله)
            word_text = "".join(word[1:])
            label = self.label_padding(word_text, num_tokens)
            
            if label is None or len(label) != self.output_max_len:
                continue
            
            wids.append(wid)
            idxs.append(idx)
            imgs.append(img)
            img_widths.append(img_width)
            labels.append(label)

        if len(list(set(wids))) != 1:
            print("Error! writer id differs")
            exit()

        final_wid = wid_idx_num
        num_imgs = len(imgs)
        
        if num_imgs >= EXTRA_CHANNEL:
            final_img = np.stack(imgs[:EXTRA_CHANNEL], axis=0)
            final_idx = idxs[:EXTRA_CHANNEL]
            final_img_width = img_widths[:EXTRA_CHANNEL]
            final_label = np.array(labels[:EXTRA_CHANNEL])
        else:
            final_idx = idxs
            final_img = imgs
            final_img_width = img_widths
            final_label = labels

            while len(final_img) < EXTRA_CHANNEL:
                num_cp = EXTRA_CHANNEL - len(final_img)
                final_idx = final_idx + idxs[:num_cp]
                final_img = final_img + imgs[:num_cp]
                final_img_width = final_img_width + img_widths[:num_cp]
                final_label = final_label + labels[:num_cp]
            
            final_img = np.stack(final_img, axis=0)
            final_label = np.array(final_label)

        _id = np.random.randint(EXTRA_CHANNEL)
        img_xt = final_img[_id : _id + 1]
        
        if self.oov:
            label_xt = np.random.choice(text_corpus)
            label_xt = np.array(self.label_padding(label_xt, num_tokens))
            label_xt_swap = np.random.choice(text_corpus)
            label_xt_swap = np.array(self.label_padding(label_xt_swap, num_tokens))
        else:
            label_xt = final_label[_id]
            label_xt_swap = self.new_ed1(label_xt)

        final_idx = np.delete(final_idx, _id, axis=0)
        final_img = np.delete(final_img, _id, axis=0)
        final_img_width = np.delete(final_img_width, _id, axis=0)
        final_label = np.delete(final_label, _id, axis=0)

        return (
            "src",
            final_wid,
            final_idx,
            final_img,
            final_img_width,
            final_label,
            img_xt,
            label_xt,
            label_xt_swap,
        )

    def read_image_single(self, file_name):
        """خواندن تصویر کلمه فارسی"""
        url = os.path.join(img_base, file_name + ".png")
        img = cv2.imread(url, 0)

        if img is None and os.path.exists(url):
            # image is present but corrupted
            return np.zeros((IMG_HEIGHT, IMG_WIDTH)), 0

        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(
            img,
            (int(img.shape[1] * rate) + 1, IMG_HEIGHT),
            interpolation=cv2.INTER_CUBIC,
        )

        # 🟢 --- اعمال Augmentation ---
        # فقط بر روی داده‌های آموزشی و به صورت احتمالی اعمال می‌شود
        if self.is_train and self.transform:
            # Albumentations انتظار دارد تصویر uint8 باشد، پس موقتا تبدیل می‌کنیم
            augmented = self.transform(image=img) # img is already uint8 here
            img = augmented['image']
        # ---------------------------

        img = img / 255.0  # 0-255 -> 0-1
        img = 1.0 - img
        img_width = img.shape[-1]

        if img_width > IMG_WIDTH:
            outImg = img[:, :IMG_WIDTH]
            img_width = IMG_WIDTH
        else:
            outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype="float32")
            outImg[:, :img_width] = img
        
        outImg = outImg.astype("float32")
        mean = 0.5
        std = 0.5
        outImgFinal = (outImg - mean) / std
        return outImgFinal, img_width

    def label_padding(self, label_text, num_tokens):
        """تبدیل رشته به لیست اندیس‌ها با padding"""
        label_indices = []
        for char in label_text:
            if char in letter2index:
                label_indices.append(letter2index[char])
        
        # برش کلمات طولانی
        if len(label_indices) > MAX_CHARS:
            label_indices = label_indices[:MAX_CHARS]
        
        # اضافه کردن offset
        label_indices = [idx + num_tokens for idx in label_indices]
        
        # اضافه کردن GO و END
        final_label = [tokens["GO_TOKEN"]] + label_indices + [tokens["END_TOKEN"]]
        
        # Padding
        padding_length = self.output_max_len - len(final_label)
        if padding_length > 0:
            final_label.extend([tokens["PAD_TOKEN"]] * padding_length)
        elif padding_length < 0:
            final_label = final_label[:self.output_max_len]
        
        return final_label


def loadData(oov):
    """بارگذاری داده‌های آموزش و تست ABAN"""
    gt_tr = src
    gt_te = tar

    with open(gt_tr, "r", encoding='utf-8') as f_tr:
        data_tr = f_tr.readlines()
        data_tr = [i.strip().split(" ") for i in data_tr]
        tr_dict = dict()
        for i in data_tr:
            wid = i[0].split(",")[0]
            if wid not in tr_dict.keys():
                tr_dict[wid] = [i]
            else:
                tr_dict[wid].append(i)
        
        new_tr_dict = dict()
        if CREATE_PAIRS:
            create_pairs(tr_dict)
        for k in tr_dict.keys():
            new_tr_dict[wid2label_tr[k]] = tr_dict[k]

    with open(gt_te, "r", encoding='utf-8') as f_te:
        data_te = f_te.readlines()
        data_te = [i.strip().split(" ") for i in data_te]
        te_dict = dict()
        for i in data_te:
            wid = i[0].split(",")[0]
            if wid not in te_dict.keys():
                te_dict[wid] = [i]
            else:
                te_dict[wid].append(i)
        
        new_te_dict = dict()
        if CREATE_PAIRS:
            create_pairs(te_dict)
        for k in te_dict.keys():
            new_te_dict[wid2label_te[k]] = te_dict[k]

    data_train = ABAN_words(new_tr_dict, oov, is_train=True)
    data_test = ABAN_words(new_te_dict, oov, is_train=False)
    return data_train, data_test


def create_pairs(ddict):
    """ایجاد جفت‌های (label, writer_id)"""
    num = len(ddict.keys())
    label2wid = list(zip(range(num), ddict.keys()))
    print(label2wid)


if __name__ == "__main__":
    pass