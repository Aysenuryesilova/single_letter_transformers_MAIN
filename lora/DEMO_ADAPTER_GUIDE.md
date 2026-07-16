# `demo.ipynb`'deki Modele LoRA Adaptörü Takma — Basit Rehber

Bu rehber, `demo.ipynb` içinde zaten eğitip yüklediğin küçük `TinyQwen`
modeline (`qwen3/tiny_qwen.pt`, 197 parametre), `lora/` klasöründeki hazır
kodla minik bir LoRA adaptörü takıp yapısını yazdırmak için 7 basit adımı
anlatır. LoRA'nın teorisi için `lora_turkce_anlatim.md` dosyasına bakabilirsin
— burada sadece **pratik akış** var.

Önkoşul: notebook'ta `model`, `tok` ve `ckpt` değişkenleri zaten yüklenmiş
olmalı (bkz. `demo.ipynb`'nin ilk hücreleri).

---

## Adım 1 — `lora/` klasörünü import path'ine ekle

```python
sys.path.insert(0, "lora")
from lora import LoRAConfig
from inject import (inject, set_adapters, merge_adapters, print_parameter_report,
                    print_linear_names, save_adapter)
```

## Adım 2 — Hangi katmanlar adapte edilebilir?

```python
print_linear_names(model)
```

Bu, modeldeki her `nn.Linear` katmanının adını ve boyutunu listeler
(`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj`...). `lm_head` hariç tutulur çünkü embedding ile ağırlık
paylaşımı (weight tying) var.

## Adım 3 — Adaptörü tak ve yapıyı yazdır

```python
lcfg = LoRAConfig(r=4, alpha=8.0, targets=("q_proj", "v_proj"))
adapted_layers = inject(model, lcfg, method="lora")
print("adapte edilen katmanlar:", adapted_layers)

model   # <- adaptörün yapısı: Linear yerine LoRALinear(base, lora_A, lora_B)
```

`model`'i yazdırdığında, hedeflenen `q_proj`/`v_proj` katmanlarının artık
`LoRALinear(base=Linear(...), dropout=...)` olduğunu, içinde de eğitilecek
`lora_A` ve `lora_B` parametrelerinin bulunduğunu görürsün.

## Adım 4 — Eğitilebilir / toplam parametre oranı

```python
print_parameter_report(model)
```

Örnek: `trainable: 896 / 20,480 (%4.38)` — yani modelin sadece küçük bir
kısmı eğitilecek, geri kalan her şey donuk kalıyor.

## Adım 5 — Minik eğitim verisi hazırla ve sadece adaptörü eğit

```python
letter = "z"
raw = open("data/temiz_isimler.txt", encoding="utf-8").read().split("\n")
names = [n for n in raw if n and n[0] == letter]
text = "\n" + "\n".join(names) + "\n"
data = torch.tensor(tok.encode(text), dtype=torch.long)

BLOCK_SIZE, BATCH_SIZE, STEPS, LR = 16, 64, 300, 5e-3
torch.manual_seed(0)

def get_batch(data):
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x, y

opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
model.train()
for step in range(1, STEPS + 1):
    x, y = get_batch(data)
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 50 == 0 or step == 1:
        print(f"step {step:4d}  loss {loss.item():.4f}")
model.eval()
```

`optimizer` sadece `requires_grad=True` olan parametreleri (yani adaptörü)
günceller; base model hiç değişmez.

## Adım 6 — Adaptör açık/kapalı karşılaştırması

```python
@torch.no_grad()
def sample(n=20):
    start = torch.full((n, 1), tok.eos_id, dtype=torch.long)
    out = model.generate(start, max_new_tokens=model.cfg.max_seq_len,
                         temperature=0.8, eos_id=tok.eos_id)
    names = [tok.decode(r[1:]).split("\n")[0] for r in out.tolist()]
    return [nm for nm in names if nm]

set_adapters(model, True)
print("adaptör AÇIK :", sample())
set_adapters(model, False)
print("adaptör KAPALI (orijinal model):", sample())
set_adapters(model, True)
```

Adaptör açıkken üretilen isimlerin `letter` ile başlama oranı belirgin
şekilde artmalı; kapalıyken model orijinal (genel) davranışına döner.

## Adım 7 — Adaptörü kaydet ve gerçek tensör yapısını yazdır

```python
adapter_path = f"qwen3/adapter_{letter}.pt"
save_adapter(model, adapter_path, "lora", lcfg, arch="qwen3")

adapter_ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)
for name, tensor in adapter_ckpt["adapter"].items():
    print(f"{name:30s} {tuple(tensor.shape)}")
```

Çıktı, her adapte edilmiş katman için `lora_A` (`r x in`) ve `lora_B`
(`out x r`) tensörlerinin şeklini gösterir — adaptörün kayıtlı dosyadaki tam
karşılığı budur.

## Adım 8 — LoRA matrislerinin orijinal ağırlığa adım adım eklenmesi

Her adapte edilmiş katmanda (örn. `layers.0.attn.q_proj`, `W`: `[32, 32]`) iki
küçük matris var:

- **`lora_A`** şekli `[r, in] = [4, 32]` — girdiyi 32 boyuttan 4 boyutlu bir
  "dar boğaza" sıkıştırır.
- **`lora_B`** şekli `[out, r] = [32, 4]` — bu 4 boyutlu ara sonucu tekrar 32
  boyuta genişletir.

Birleşme iki matematiksel olarak **birebir eşdeğer** yoldan anlaşılabilir:

**A) Ağırlık düzeyinde (merge — kalıcı birleştirme):**

```text
delta_W = scale * (B @ A)      # [32,4] @ [4,32] -> [32,32], W ile AYNI boyut
W_yeni  = W_orijinal + delta_W
```

(`scale = alpha / r`; burada `8.0 / 4 = 2.0`.)

**B) Girdi düzeyinde (forward — merge yapmadan, aynı sonucu üretir):**

```text
h      = x @ A.T            # [1,32] @ [32,4] -> [1,4]     (sıkıştır)
y_lora = scale * (h @ B.T)  # [1,4]  @ [4,32] -> [1,32]    (genişlet + ölçekle)
y      = (x @ W_orijinal.T) + y_lora
```

Kod olarak (Adım 7'de kaydettiğin `qwen3/adapter_z.pt` dosyasındaki gerçek
sayılarla):

```python
# tek katman: elle delta_W hesapla
W = ckpt["model"]["layers.0.attn.q_proj.weight"]          # orijinal, hiç değişmedi
saved = torch.load(adapter_path, map_location="cpu", weights_only=False)
A = saved["adapter"]["layers.0.attn.q_proj.lora_A"]        # [4, 32]
B = saved["adapter"]["layers.0.attn.q_proj.lora_B"]        # [32, 4]
scale = saved["cfg"].alpha / saved["cfg"].r

delta = scale * (B @ A)     # [32,32]
W_new = W + delta

# gerçek bir girdiyle (örn. 'z' harfinin embedding'i) iki yolu karşılaştır
zid = tok.encode("z")[0]
x = ckpt["model"]["embed_tokens.weight"][zid].unsqueeze(0)   # [1, 32]

y_base = x @ W.T
h      = x @ A.T
y_lora = scale * (h @ B.T)
y_stepwise = y_base + y_lora      # A) adım adım
y_merged   = x @ W_new.T          # B) tek seferde, W_yeni ile

torch.allclose(y_stepwise, y_merged, atol=1e-5)   # -> True
```

Bu küçük modelde (`tiny_qwen.pt`, r=4) bunu çalıştırınca gerçekten `True`
dönüyor: iki yol da birebir aynı sayıları üretiyor.

## Adım 9 — `merge_adapters` ile tüm katmanları birden birleştirmek

Adım 8'de tek bir katman için elle yaptığımız `delta_W = scale * (B @ A)`
hesabını, `inject.py` içindeki `merge_adapters(model)` fonksiyonu **adapte
edilmiş her katman için otomatik olarak** yapar:

```python
def merge_adapters(model):
    for m in model.modules():
        if hasattr(m, "merge"):
            m.merge()
```

Her `LoRALinear.merge()` çağrısı:

1. Kendi `base.weight`'ini `+= delta_weight()` ile kalıcı olarak günceller.
2. `enabled = False` yapar, böylece bir daha `lora_A`/`lora_B` toplanıp
   düzeltme iki kere eklenmez.

Kullanımı:

```python
q_proj_layer = model.layers[0].attn.q_proj
print(q_proj_layer.enabled)                       # True
print(torch.allclose(q_proj_layer.base.weight, W)) # True -- hâlâ orijinal

xb, _ = get_batch(data)
before_logits = model(xb)[0]

merge_adapters(model)                              # <- tüm katmanları birleştir

print(q_proj_layer.enabled)                        # False
print(torch.allclose(q_proj_layer.base.weight, W_new, atol=1e-5))  # True

after_logits = model(xb)[0]
max_diff = (before_logits - after_logits).abs().max().item()
print(max_diff)   # ~1e-5 seviyesinde, yani pratikte 0 -- model tamamen aynı davranıyor
```

Yani `merge_adapters` sonrası model artık sıradan bir `TinyQwen` gibi
çalışır: çıktı merge öncesiyle birebir aynıdır, ama artık ayrı bir adapter
yolu çalıştırmaya gerek kalmaz (inference ucuzlar).

⚠️ Bu işlem `model` nesnesini **bellekte kalıcı olarak** değiştirir. Adaptörü
tekrar ayrı ayrı açıp kapatmak istersen modeli diskten yeniden yükleyip Adım
3'ü tekrar çalıştırman gerekir — diskteki `tiny_qwen.pt` ve
`qwen3/adapter_z.pt` dosyaları bu işlemden etkilenmez.

---

## Özet akış

```text
1. model dondurulur (zaten eğitilmiş, değişmeyecek)
2. inject() ile q_proj/v_proj'a LoRALinear takılır (lora_A, lora_B eklenir)
3. sadece lora_A/lora_B eğitilir (~%4 parametre)
4. adaptör açık/kapalı ile davranış farkı gözlemlenir
5. save_adapter() ile birkaç KB'lık adaptör dosyası kaydedilir
```

Diğer yöntemleri denemek istersen (`method="rslora" | "dora" | "vera" | "pissa"`),
sadece Adım 3'teki `inject(..., method=...)` çağrısını değiştirmen yeterli —
gerisi aynı kalır.
