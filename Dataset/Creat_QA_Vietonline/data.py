from datasets import load_dataset
import os

# Tải dataset
ds = load_dataset("VLUS06/VietOnlineNews")

# Tạo thư mục lưu trữ nếu chưa có
save_dir = os.path.join(os.path.dirname(__file__), "VietOnlineNews_CSV")
os.makedirs(save_dir, exist_ok=True)

# Xuất từng split (train, validation, test) ra file CSV riêng biệt
for split in ds.keys():
    csv_path = os.path.join(save_dir, f"{split}.csv")
    ds[split].to_csv(csv_path, index=False)
    print(f"Đã lưu {split} vào {csv_path}")