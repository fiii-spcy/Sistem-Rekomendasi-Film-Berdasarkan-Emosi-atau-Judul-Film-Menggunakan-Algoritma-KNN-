import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# =============================================================
# === PENTING: Konfigurasi Matplotlib dan Impor Embedding ===
# =============================================================
import matplotlib
matplotlib.use('TkAgg') 

import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
import requests 
from PIL import Image, ImageTk 
from io import BytesIO 

# Impor yang dibutuhkan untuk Embedding Plot ke Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk 
# =============================================================


# --- 1. DEFINISI PEMETAAN DAN MODEL TRAINING ---

def map_genre_to_class(genre, title):
    """
    Memetakan genre/judul film ke salah satu dari 5 kelas emosional/genre utama.
    """
    g = str(genre).lower()
    t = str(title).lower()
    
    # Kelas 1: COMEDY / KEGEMBIRAAN
    if 'comedy' in g or 'musical' in g or 'animation' in g:
        return 1, "lucu bahagia komedi tawa senang gembira pesta"  
    # Kelas 2: ACTION / ADVENTURE / KEMARAHAN
    elif 'action' in g or 'adventure' in g or 'fantasy' in g:
        return 2, "perang marah tembak aksi pertarungan petualangan berani"
    # Kelas 3: DRAMA / EMOSIONAL / SEDIH/SAKIT HATI
    elif 'drama' in g or 'romantic' in g: 
        return 3, "sedih nangis emosi hati drama romantis sakit galau"
    # Kelas 4: HORROR / MYSTERY / TAKUT/CEMAS
    elif 'horror' in g or 'mystery' in g or 'thriller' in g:
        return 4, "takut cemas misteri tegang horor hantu gelap"
    # Kelas 5: FAMILY / LAINNYA / NETRAL
    else:
        return 5, "keluarga netral pendidikan sejarah biografi"

KELAS_MAP = {
    1: "COMEDY / KEGEMBIRAAN",
    2: "ACTION / ADVENTURE / KEMARAHAN",
    3: "DRAMA / EMOSIONAL / SEDIH/SAKIT HATI",
    4: "HORROR / MYSTERY / TAKUT/CEMAS",
    5: "FAMILY / LAINNYA / NETRAL"
}
KELAS_LABELS = list(KELAS_MAP.values()) 


def train_knn_multiclass_classifier():
    """Melatih model KNN Klasifikasi Multikelas dan memuat data film."""
    try:
        df = pd.read_csv("disney_movies_with_posters.csv").copy() 
        if 'poster_url' not in df.columns:
            df['poster_url'] = None 
    except FileNotFoundError:
        messagebox.showerror("Error File", "File 'disney_movies_with_posters.csv' tidak ditemukan. Harap jalankan script penambah poster terlebih dahulu!")
        return None, None, None, None, None 

    results = df.apply(lambda row: map_genre_to_class(row['genre'], row['movie_title']), axis=1, result_type='expand')
    df['target_class'] = results[0]
    df['keyword_boost'] = results[1]

    df['features'] = df['genre'].fillna('') + " " + df['movie_title'].fillna('') + " " + df['keyword_boost']
    df['features'] = df['features'].str.lower()
    df['movie_title_lower'] = df['movie_title'].fillna('').str.lower()

    X = df['features']
    y = df['target_class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    tfidf = TfidfVectorizer(stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    K = 7 
    knn_model = KNeighborsClassifier(n_neighbors=K, metric='cosine')
    knn_model.fit(X_train_tfidf, y_train)
    
    print(f"Model KNN Klasifikasi Multikelas ({len(KELAS_MAP)} kelas) berhasil dilatih menggunakan {len(df)} data.")
    return knn_model, tfidf, X_test, y_test, df 


# --- 2. FUNGSI EVALUASI PLOTTING ---

def calculate_and_plot_metrics():
    """Menghitung metrik (CM) dan langsung memunculkan plot visualisasi."""
    if knn_model is None or X_test_eval is None:
        lbl_eval_status.config(text="Status: ERROR - Model belum dilatih atau data tidak ada.", fg="red")
        return
        
    lbl_eval_status.config(text="Status: Menghitung Matriks Kebingungan dan Metrik...", fg="yellow")
    root.update_idletasks() 
    
    X_test_tfidf = tfidf_vectorizer.transform(X_test_eval)
    y_pred = knn_model.predict(X_test_tfidf)
    
    global cm_global, y_test_global, y_pred_global 
    cm_global = confusion_matrix(y_test_eval, y_pred, labels=list(KELAS_MAP.keys())) 
    y_test_global = y_test_eval
    y_pred_global = y_pred
    
    accuracy = knn_model.score(X_test_tfidf, y_test_eval)

    # Cetak Metrik ke Terminal (PENTING: Pastikan Anda melihat konsol/terminal ini!)
    print("\n=======================================================")
    print("=== HASIL EVALUASI MODEL KNN KLASIFIKASI MULTIKELAS ===")
    print("=======================================================")
    print(f"Akurasi Model (Overall): {accuracy:.4f}")
    print(f"\nMatriks Kebingungan (CM {len(KELAS_MAP)}x{len(KELAS_MAP)}):\n(Baris = Aktual | Kolom = Prediksi)")
    print(cm_global)
    print("=======================================================")
    
    show_plots()


def show_plots():
    """Menampilkan Plot Visualisasi Matplotlib dalam jendela Toplevel baru."""
    try:
        cm = cm_global
        y_test = y_test_global
        y_pred = y_pred_global
        if cm is None:
            lbl_eval_status.config(text="Status: ERROR - Silakan tekan PLOT VISUALISASI terlebih dahulu!", fg="red")
            return

    except NameError:
        lbl_eval_status.config(text="Status: ERROR - Silakan tekan PLOT VISUALISASI terlebih dahulu!", fg="red")
        return

    try:
        # --- 1. Buat Jendela Baru (Toplevel) ---
        plot_window = tk.Toplevel(root)
        plot_window.title("Visualisasi Kinerja Model")
        plot_window.geometry("1000x650")
        plot_window.configure(bg="#e0e0e0")
        
        # --- 2. Buat Figure Matplotlib ---
        # PERBAIKAN PLOT: Perbesar figsize agar label tidak berdempet
        fig, axes = plt.subplots(1, 2, figsize=(16, 7)) 
        
        axes[0].scatter(range(len(y_test)), y_test, color='green', label='Aktual (y_test)')
        axes[0].scatter(range(len(y_pred)), y_pred, marker='x', color='red', label='Prediksi (y_pred)')
        axes[0].set_yticks(list(KELAS_MAP.keys()))
        axes[0].set_yticklabels(KELAS_LABELS)
        axes[0].set_title('Aktual vs Prediksi Kelas Film')
        axes[0].set_xlabel('Indeks Sampel Data Testing')
        axes[0].set_ylabel('Kelas Prediksi')
        axes[0].grid(True, alpha=0.5)
        axes[0].legend()
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=True, ax=axes[1],
                    xticklabels=KELAS_LABELS, yticklabels=KELAS_LABELS)
        
        axes[1].set_title(f'Confusion Matrix (Heatmap) {len(KELAS_MAP)}x{len(KELAS_LABELS)}')
        axes[1].set_xlabel('Prediksi')
        axes[1].set_ylabel('Aktual')
        
        # PERBAIKAN PLOT: Rotasi label sumbu X pada Confusion Matrix
        plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor") 

        fig.suptitle("Visualisasi Kinerja Model KNN Klasifikasi Multikelas", fontsize=16, color='black')
        fig.patch.set_facecolor('#e0e0e0') 
        plt.tight_layout() # Mengatur tata letak agar label pas

        # --- 3. Embed Figure ke Toplevel Window ---
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Tambahkan Toolbar 
        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()
        
        # Close the Matplotlib figure to free memory after embedding
        plt.close(fig)

        lbl_eval_status.config(text="Status: Evaluasi Selesai, Plot Visualisasi ditampilkan dalam Jendela Baru.", fg="#00ff88")
        
    except Exception as e:
        lbl_eval_status.config(text=f"Status: ERROR saat membuat Plot: {e}", fg="red")
        print(f"ERROR Matplotlib/Tkinter (Embedding Gagal): {e}")


# --- 3. LOGIKA GUI DAN PREDIKSI ---

knn_model, tfidf_vectorizer, X_test_eval, y_test_eval, df_movies = train_knn_multiclass_classifier()

if knn_model is None:
    exit()

cm_global = None
y_test_global = None
y_pred_global = None

PLACEHOLDER_IMAGE_URL = "https://via.placeholder.com/100x150?text=No+Poster" 
placeholder_image = None 
photo_images = {} 


def map_genre_to_class(genre, title):
    # ... (fungsi ini tetap sama seperti sebelumnya) ...
    g = str(genre).lower()
    t = str(title).lower()
    if 'comedy' in g or 'musical' in g or 'animation' in g:
        return 1, "lucu bahagia komedi tawa senang gembira pesta"  
    elif 'action' in g or 'adventure' in g or 'fantasy' in g:
        return 2, "perang marah tembak aksi pertarungan petualangan berani"
    elif 'drama' in g or 'romantic' in g: 
        return 3, "sedih nangis emosi hati drama romantis sakit galau"
    elif 'horror' in g or 'mystery' in g or 'thriller' in g:
        return 4, "takut cemas misteri tegang horor hantu gelap"
    else:
        return 5, "keluarga netral pendidikan sejarah biografi"


def display_multiple_posters(poster_urls):
    """Mengunduh dan menampilkan hingga 3 poster (Mode Emosional)."""
    global photo_images
    
    for i in range(3):
        lbl_posters[i].config(image='')
    photo_images.clear() 

    for i, poster_url in enumerate(poster_urls):
        if i >= 3:
            break

        img = None
        try:
            if pd.isna(poster_url) or not poster_url:
                raise ValueError("URL Poster tidak valid atau kosong.")
            
            response = requests.get(poster_url)
            response.raise_for_status() 
            img_data = response.content
            img = Image.open(BytesIO(img_data))
        except (requests.exceptions.RequestException, ValueError, IOError) as e:
            if placeholder_image:
                img = placeholder_image
            else:
                img = Image.new('RGB', (100, 150), color = 'gray')
                
        img = img.resize((100, 150), Image.LANCZOS) 
        photo_images[i] = ImageTk.PhotoImage(img)
        lbl_posters[i].config(image=photo_images[i])
        lbl_posters[i].image = photo_images[i] 

def predict_user_preference_and_eval():
    """Memprediksi input pengguna dan menampilkan 3 poster rekomendasi."""
    user_input = entry_input_emotion.get().lower().strip()
    
    lbl_prediksi.config(text="")
    lbl_rekomendasi.config(text="")
    display_multiple_posters([]) 
    
    if not user_input:
        lbl_prediksi.config(text="INPUT KOSONG", fg="red")
        lbl_rekomendasi.config(text="Masukkan kondisi psikologis/emosi Anda.", fg="#7788aa")
        lbl_eval_status.config(text="Status: Input emosional kosong.", fg="red")
        return

    user_vector = tfidf_vectorizer.transform([user_input])
    predicted_class = knn_model.predict(user_vector)[0]
    predicted_label = KELAS_MAP.get(predicted_class, "TIDAK DIKETAHUI")
    
    hasil_text = f"PREDIKSI KELAS:\n{predicted_label}"
    
    recommended_movies = df_movies[df_movies['target_class'] == predicted_class]
    
    poster_urls_to_display = []
    recommended_titles_text = "Film Rekomendasi:\n"
    
    if not recommended_movies.empty:
        num_to_sample = min(3, len(recommended_movies))
        random_movies = recommended_movies.sample(n=num_to_sample)
        
        for index, movie in random_movies.iterrows():
            poster_urls_to_display.append(movie['poster_url'])
            recommended_titles_text += f"• {movie['movie_title']}\n"
            
        display_multiple_posters(poster_urls_to_display)
    else:
        display_multiple_posters([]) 
        recommended_titles_text += "• Tidak ada film yang ditemukan untuk kelas ini."
        
    lbl_prediksi.config(text=hasil_text, fg="#00ff88", justify=tk.LEFT, anchor='w')
    lbl_rekomendasi.config(text=recommended_titles_text, fg="#00d4ff", justify=tk.LEFT, anchor='w')
        
    lbl_eval_status.config(text="*Status: Prediksi Selesai. Metrik CM telah dicetak ke Terminal. Tekan 'PLOT VISUALISASI' untuk visualisasi.", fg="#00ff88")


def search_by_title():
    """Logika pencarian film berdasarkan judul/kata kunci."""
    search_input = entry_input_title.get().lower().strip()
    
    lbl_output_title.config(text="")
    lbl_poster_search.config(image='')
    
    if not search_input:
        lbl_output_title.config(text="Masukkan Judul Film atau Kata Kunci.", fg="red", justify=tk.LEFT)
        lbl_search_status.config(text="Status: Input pencarian kosong.", fg="red")
        return
        
    lbl_search_status.config(text="Status: Mencari film...", fg="yellow")
    root.update_idletasks()
    
    results = df_movies[df_movies['movie_title_lower'].str.contains(search_input, na=False)]
    
    if results.empty:
        lbl_output_title.config(text=f"Film dengan judul/kata kunci '{search_input}' tidak ditemukan.", fg="red", justify=tk.LEFT)
        lbl_search_status.config(text="Status: Film tidak ditemukan.", fg="red")
        display_poster_search(None) 
    else:
        output_text = f"Ditemukan {len(results)} film (Max 3 Ditampilkan):\n\n"
        first_movie = results.iloc[0]
        display_poster_search(first_movie['poster_url'])

        for index, row in results.head(3).iterrows(): 
            predicted_label = KELAS_MAP.get(row['target_class'], 'N/A')
            output_text += f"• {row['movie_title']}\n  (Genre: {row['genre']}, Kelas: {predicted_label})\n\n"
            
        lbl_output_title.config(text=output_text, fg="white", justify=tk.LEFT, anchor='w')
        lbl_search_status.config(text="Status: Pencarian Selesai.", fg="#00ff88")

def display_poster_search(poster_url):
    """Mengunduh dan menampilkan poster untuk hasil pencarian (Mode Judul)."""
    global photo_image_search
    try:
        if pd.isna(poster_url) or not poster_url:
            raise ValueError("URL Poster tidak valid atau kosong.")
        
        response = requests.get(poster_url)
        response.raise_for_status()
        img_data = response.content
        img = Image.open(BytesIO(img_data))
    except (requests.exceptions.RequestException, ValueError, IOError) as e:
        if placeholder_image:
            img = placeholder_image.copy()
            img = img.resize((150, 225), Image.LANCZOS)
        else:
            img = Image.new('RGB', (150, 225), color = 'gray')
            
    img = img.resize((150, 225), Image.LANCZOS)
    photo_image_search = ImageTk.PhotoImage(img)
    lbl_poster_search.config(image=photo_image_search)
    lbl_poster_search.image = photo_image_search


# --- 4. GUI TKINTER ---

root = tk.Tk()
root.title("Sistem Klasifikasi & Pencarian Film")
root.geometry("1000x800") 
root.configure(bg="#0d1b2a") 

try:
    response = requests.get(PLACEHOLDER_IMAGE_URL)
    response.raise_for_status()
    img_data = response.content
    placeholder_image = Image.open(BytesIO(img_data)).resize((100, 150), Image.LANCZOS) 
except Exception as e:
    placeholder_image = Image.new('RGB', (100, 150), color = 'gray')

# --- Header dan Tombol Evaluasi ---
header_frame = tk.Frame(root, bg="#0d1b2a")
header_frame.pack(fill="x", pady=(10, 5))

tk.Label(header_frame, text="SISTEM PREDIKSI & PENCARIAN FILM", font=("Arial Black", 24, "bold"),
          fg="#00d4ff", bg="#0d1b2a").pack(side="left", padx=20)

btn_show_plot = tk.Button(header_frame, text="PLOT VISUALISASI", 
                          font=("Arial", 10, "bold"), bg="#ff7f50", fg="black", 
                          command=calculate_and_plot_metrics, cursor="hand2", relief="flat", padx=5)
btn_show_plot.pack(side="right", padx=20)


# --- Container untuk Dua Mode ---
main_container = tk.Frame(root, bg="#0d1b2a")
main_container.pack(pady=10, padx=20, fill="both", expand=True)

# --- Frame untuk Mode 1: KLASIFIKASI EMOSIONAL ---
emotion_mode_frame = tk.LabelFrame(main_container, text="MODE 1: KLASIFIKASI EMOSIONAL / PSIKOLOGIS", 
                                  font=("Arial", 13, "bold"), fg="#e0e0e0", bg="#1b263b", bd=2, relief="groove", 
                                  padx=15, pady=15, labelanchor='n') 
emotion_mode_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
main_container.grid_columnconfigure(0, weight=1)
main_container.grid_rowconfigure(0, weight=1)

# Input Emosi
input_emotion_frame = tk.Frame(emotion_mode_frame, bg="#1b263b")
input_emotion_frame.pack(pady=(10, 15))

tk.Label(input_emotion_frame, text="Ketik Kondisi Psikologis/Emosi:", font=("Arial", 12), 
          fg="#aaa", bg="#1b263b").pack(side="left", padx=5)

entry_input_emotion = tk.Entry(input_emotion_frame, font=("Arial", 14), width=25, justify="left",
                               bg="#0d1b2a", fg="white", relief="flat", insertbackground="white")
entry_input_emotion.pack(side="left", padx=5)
entry_input_emotion.insert(0, "saya lagi sedih rekomendasi film") 

# PERBAIKAN TOMBOL FINAL V3: Menggunakan padx tinggi
btn_predict = tk.Button(input_emotion_frame, text="PREDIKSI & EVALUASI", font=("Arial", 12, "bold"), 
                       bg="#00d4ff", fg="#0d1b2a", command=predict_user_preference_and_eval, 
                       cursor="hand2", padx=30, relief="flat") # <-- Ditingkatkan ke padx=30
btn_predict.pack(side="left", padx=5)

# Output Emosi
output_emotion_display_frame = tk.Frame(emotion_mode_frame, bg="#0d1b2a", bd=1, relief="solid", padx=10, pady=10)
output_emotion_display_frame.pack(pady=10, fill="x", expand=True)


# Frame untuk 3 Poster
poster_frame = tk.Frame(output_emotion_display_frame, bg="#0d1b2a")
poster_frame.pack(side="left", padx=10, pady=5)

lbl_posters = [] 
for i in range(3):
    lbl = tk.Label(poster_frame, bg="#0d1b2a")
    lbl.pack(side="left", padx=5)
    lbl_posters.append(lbl)

# Teks Prediksi Emosi (kanan)
text_output_emotion_frame = tk.Frame(output_emotion_display_frame, bg="#0d1b2a")
text_output_emotion_frame.pack(side="right", fill="both", expand=True, padx=(0,10))

lbl_prediksi = tk.Label(text_output_emotion_frame, text="[HASIL KLASIFIKASI]", font=("Arial", 16, "bold"), 
                        fg="#00ff88", bg="#0d1b2a", justify=tk.LEFT, wraplength=200, anchor='w') 
lbl_prediksi.pack(pady=5, anchor='w')

lbl_rekomendasi = tk.Label(text_output_emotion_frame, text="[Contoh Film Rekomendasi]", font=("Arial", 10), 
                           fg="#00d4ff", bg="#0d1b2a", justify=tk.LEFT, wraplength=200, anchor='w') 
lbl_rekomendasi.pack(pady=(0, 5), anchor='w')

# Status Evaluasi
lbl_eval_status = tk.Label(emotion_mode_frame, text="*Status: Tekan 'PREDIKSI & EVALUASI' untuk rekomendasi. Gunakan tombol atas untuk Plot Visualisasi.", 
                  font=("Arial", 9), fg="#7788aa", bg="#1b263b", wraplength=400)
lbl_eval_status.pack(pady=5)


# --- Frame untuk Mode 2: PENCARIAN BERDASARKAN JUDUL FILM ---
search_mode_frame = tk.LabelFrame(main_container, text="MODE 2: PENCARIAN BERDASARKAN JUDUL FILM", 
                                 font=("Arial", 13, "bold"), fg="#e0e0e0", bg="#1b263b", bd=2, relief="groove", 
                                 padx=15, pady=15, labelanchor='n')
search_mode_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
main_container.grid_columnconfigure(1, weight=1)
main_container.grid_rowconfigure(0, weight=1)


# Input Judul
input_title_frame = tk.Frame(search_mode_frame, bg="#1b263b")
input_title_frame.pack(pady=(10, 15))

tk.Label(input_title_frame, text="Ketik Judul Film/Kata Kunci:", font=("Arial", 12), 
          fg="#aaa", bg="#1b263b").pack(side="left", padx=5)

entry_input_title = tk.Entry(input_title_frame, font=("Arial", 14), width=25, justify="left",
                             bg="#0d1b2a", fg="white", relief="flat", insertbackground="white")
entry_input_title.pack(side="left", padx=5)
entry_input_title.insert(0, "Muppets Most Wanted")

# PERBAIKAN TOMBOL FINAL V3: Menggunakan padx tinggi
btn_search = tk.Button(input_title_frame, text="CARI FILM", font=("Arial", 12, "bold"), 
                       bg="#00d4ff", fg="#0d1b2a", command=search_by_title, 
                       cursor="hand2", padx=30, relief="flat") # <-- Ditingkatkan ke padx=30
btn_search.pack(side="left", padx=5)

# Output Judul
output_title_display_frame = tk.Frame(search_mode_frame, bg="#0d1b2a", bd=1, relief="solid", padx=10, pady=10)
output_title_display_frame.pack(pady=10, fill="x", expand=True)

# Label Poster Pencarian (Tetap 1)
photo_image_search = None
lbl_poster_search = tk.Label(output_title_display_frame, bg="#0d1b2a")
lbl_poster_search.pack(side="right", padx=10, pady=5)

# Teks Hasil Pencarian
lbl_output_title = tk.Label(output_title_display_frame, text="[Hasil Pencarian Judul/Kata Kunci]", font=("Arial", 11), 
                             fg="white", bg="#0d1b2a", justify=tk.LEFT, wraplength=300, anchor='w') 
lbl_output_title.pack(side="left", fill="both", expand=True, padx=(0,10), anchor='w')

# Status Pencarian
lbl_search_status = tk.Label(search_mode_frame, text="*Status: Masukkan judul film dan tekan 'CARI FILM'.", 
                  font=("Arial", 9), fg="#7788aa", bg="#1b263b", wraplength=400)
lbl_search_status.pack(pady=5)

# Inisialisasi Poster
display_multiple_posters([]) 
display_poster_search(None) 

root.mainloop()