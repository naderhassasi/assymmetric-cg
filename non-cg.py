"""
ماژول تولید داده‌های فازی برای مثال بازوی رباتیک
مقاله: روش ترکیبی CGNR با میانگین موزون تعمیم‌یافته برای دستگاه‌های خطی فازی نامتقارن
"""

import numpy as np
import os

def generate_robot_arm_data(n=1000, seed=42):
    """
    تولید داده‌های فازی برای بازوی رباتیک دو-link با روش اجزاء محدود
    
    پارامترها:
    -----------
    n : int
        تعداد نقاط گره‌ای در گسسته‌سازی FEM (پیش‌فرض: 1000)
    seed : int
        مقدار seed برای تکرارپذیری نتایج
    
    خروجی:
    -------
    A_center : ndarray
        ماتریس مرکزی (مقادیر با درجه عضویت 1)
    delta_L : ndarray
        ماتریس گستردگی چپ
    delta_R : ndarray
        ماتریس گستردگی راست
    b_center : ndarray
        بردار مرکزی سمت راست
    b_delta_L : ndarray
        بردار گستردگی چپ سمت راست
    b_delta_R : ndarray
        بردار گستردگی راست سمت راست
    params : dict
        پارامترهای فیزیکی مسئله
    """
    
    np.random.seed(seed)
    
    # ================================================
    # پارامترهای فیزیکی ربات (با عدم قطعیت فازی)
    # ================================================
    
    # Link 1
    m1_center = 5.0      # جرم مرکزی (kg)
    m1_delta_L = 0.3     # گستردگی چپ
    m1_delta_R = 0.5     # گستردگی راست
    
    # Link 2
    m2_center = 3.0
    m2_delta_L = 0.2
    m2_delta_R = 0.4
    
    # گشتاور اینرسی Link 1
    I1_center = 0.4
    I1_delta_L = 0.05
    I1_delta_R = 0.08
    
    # گشتاور اینرسی Link 2
    I2_center = 0.2
    I2_delta_L = 0.03
    I2_delta_R = 0.05
    
    # طول‌ها (قطعی)
    L1 = 0.8  # متر
    L2 = 0.6  # متر
    
    # زاویه مفصل دوم (برای محاسبه کوپلینگ)
    theta2 = np.pi/4  # 45 درجه
    
    # ================================================
    # محاسبه ماتریس سختی فازی
    # ================================================
    
    # عبارت ثابت برای درایه‌های قطری
    diag_constant_center = (m1_center * L1**2 + 
                           m2_center * (L1**2 + L2**2 + 2*L1*L2*np.cos(theta2)) +
                           I1_center + I2_center)
    
    diag_constant_delta_L = (m1_delta_L * L1**2 + 
                            m2_delta_L * (L1**2 + L2**2 + 2*L1*L2*np.cos(theta2)) +
                            I1_delta_L + I2_delta_L)
    
    diag_constant_delta_R = (m1_delta_R * L1**2 + 
                            m2_delta_R * (L1**2 + L2**2 + 2*L1*L2*np.cos(theta2)) +
                            I1_delta_R + I2_delta_R)
    
    # ایجاد ماتریس‌ها
    A_center = np.zeros((n, n))
    delta_L = np.zeros((n, n))
    delta_R = np.zeros((n, n))
    
    # درایه‌های قطری (با تغییرات ملایم در طول گره‌ها)
    for i in range(n):
        # اضافه کردن تغییرات ملایم وابسته به موقعیت گره
        variation = 1.0 + 0.1 * np.sin(2 * np.pi * i / n)
        
        A_center[i, i] = diag_constant_center * variation
        delta_L[i, i] = diag_constant_delta_L * variation
        delta_R[i, i] = diag_constant_delta_R * variation
    
    # درایه‌های غیرقطری (کوپلینگ دینامیکی)
    # ساختار تنک: فقط 10% درایه‌ها غیرصفر هستند
    coupling_strength_center = 0.15
    coupling_strength_delta_L = 0.05
    coupling_strength_delta_R = 0.08
    
    for i in range(n):
        for j in range(n):
            if i != j and np.random.random() < 0.1:  # 10% تراکم
                # وابستگی به فاصله گره‌ها
                distance_factor = np.exp(-abs(i-j)/100)
                
                A_center[i, j] = coupling_strength_center * distance_factor
                delta_L[i, j] = coupling_strength_delta_L * distance_factor
                delta_R[i, j] = coupling_strength_delta_R * distance_factor
    
    # تضمین خاصیت قطری غالب
    for i in range(n):
        row_sum_center = np.sum(np.abs(A_center[i, :])) - np.abs(A_center[i, i])
        row_sum_delta = np.sum(np.abs(delta_L[i, :])) + np.sum(np.abs(delta_R[i, :])) - (delta_L[i, i] + delta_R[i, i])
        
        # اگر قطر غالب نیست، آن را تقویت کن
        if A_center[i, i] - row_sum_center < 1.0:
            A_center[i, i] = row_sum_center + 5.0
    
    # ================================================
    # بردار سمت راست (نیروهای اعمالی فازی)
    # ================================================
    
    b_center = np.zeros(n)
    b_delta_L = np.zeros(n)
    b_delta_R = np.zeros(n)
    
    for i in range(n):
        # نیروی گرانش و گشتاور اعمالی
        b_center[i] = 10.0 * np.sin(2 * np.pi * i / n) + 5.0
        b_delta_L[i] = 0.5 + 0.1 * np.sin(i)
        b_delta_R[i] = 0.8 + 0.1 * np.cos(i)
    
    # ذخیره پارامترها
    params = {
        'm1': (m1_center, m1_delta_L, m1_delta_R),
        'm2': (m2_center, m2_delta_L, m2_delta_R),
        'I1': (I1_center, I1_delta_L, I1_delta_R),
        'I2': (I2_center, I2_delta_L, I2_delta_R),
        'L1': L1,
        'L2': L2,
        'theta2': theta2,
        'n': n
    }
    
    return A_center, delta_L, delta_R, b_center, b_delta_L, b_delta_R, params


def save_robot_arm_data(output_dir='./data/example3_robot_arm/'):
    """
    ذخیره داده‌های تولید شده در فایل‌های .npy
    """
    
    # ایجاد دایرکتوری خروجی
    os.makedirs(output_dir, exist_ok=True)
    
    # تولید داده‌ها
    A_center, delta_L, delta_R, b_center, b_delta_L, b_delta_R, params = generate_robot_arm_data()
    
    # ذخیره فایل‌ها
    np.save(os.path.join(output_dir, 'A_center.npy'), A_center)
    np.save(os.path.join(output_dir, 'delta_L.npy'), delta_L)
    np.save(os.path.join(output_dir, 'delta_R.npy'), delta_R)
    np.save(os.path.join(output_dir, 'b_center.npy'), b_center)
    np.save(os.path.join(output_dir, 'b_delta_L.npy'), b_delta_L)
    np.save(os.path.join(output_dir, 'b_delta_R.npy'), b_delta_R)
    
    # ذخیره پارامترها به صورت متنی
    with open(os.path.join(output_dir, 'parameters.txt'), 'w') as f:
        f.write("پارامترهای بازوی رباتیک دو-link\n")
        f.write("=" * 40 + "\n")
        f.write(f"m1: center={params['m1'][0]}, delta_L={params['m1'][1]}, delta_R={params['m1'][2]}\n")
        f.write(f"m2: center={params['m2'][0]}, delta_L={params['m2'][1]}, delta_R={params['m2'][2]}\n")
        f.write(f"I1: center={params['I1'][0]}, delta_L={params['I1'][1]}, delta_R={params['I1'][2]}\n")
        f.write(f"I2: center={params['I2'][0]}, delta_L={params['I2'][1]}, delta_R={params['I2'][2]}\n")
        f.write(f"L1: {params['L1']}\n")
        f.write(f"L2: {params['L2']}\n")
        f.write(f"theta2: {params['theta2']}\n")
        f.write(f"تعداد نقاط گره‌ای: {params['n']}\n")
    
    print(f"داده‌ها با موفقیت در دایرکتوری {output_dir} ذخیره شدند.")
    return output_dir


if __name__ == "__main__":
    save_robot_arm_data()