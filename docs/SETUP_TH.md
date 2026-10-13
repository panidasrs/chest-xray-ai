# โน้ตส่วนตัว Project A (ลบออกจาก repo ได้ถ้าไม่อยากให้คนอื่นเห็น)

## 1) ขึ้น GitHub

1. github.com → New repository → ชื่อ `chest-xray-ai` → Public → ไม่ต้องติ๊กอะไร → Create
2. PowerShell:

```powershell
cd $HOME\Documents\chest-xray-ai
git init -b main
git add .
git commit -m "Add data prep, training and evaluation scripts"
git remote add origin https://github.com/panidasrs/chest-xray-ai.git
git push -u origin main
```

## 2) เทรนบน Colab

1. colab.research.google.com → File → Open notebook → แท็บ GitHub → `panidasrs/chest-xray-ai` → `notebooks/train_colab.ipynb`
2. Runtime → Change runtime type → T4 GPU → Run all (~15 นาที)
3. ถ้า kagglehub ขอ login: kaggle.com → Settings → API → Create New Token แล้วรัน `import kagglehub; kagglehub.login()`
4. ได้ `results.zip` → เอาโฟลเดอร์ `results` ใส่ใน repo, `best_model.pt` เก็บใน Drive (ไม่ขึ้น GitHub)

## 3) อัปเดตผล

เอาเลขใน `results/test_results.json` ไปเติมตารางใน README แล้ว

```powershell
git add results README.md
git commit -m "Add baseline results"
git push
```
