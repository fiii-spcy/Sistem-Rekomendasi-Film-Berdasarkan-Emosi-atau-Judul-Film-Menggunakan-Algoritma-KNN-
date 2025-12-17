import pandas as pd
import requests
import time


TMDB_API_KEY = '425e3f0c6d893de7b029d5cd94f1b87d' 

SEARCH_URL = "https://api.themoviedb.org/3/search/movie"

IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w200"

def get_poster_url(movie_title):
    """Mencari judul film di TMDb dan mengembalikan URL poster lengkap."""
    
   
    if TMDB_API_KEY == 'MASUKKAN_KUNCI_API_ANDA':
        print("ERROR: Harap ganti TMDB_API_KEY dengan kunci API Anda.")
        return None
        
    params = {
        'api_key': TMDB_API_KEY,
        'query': movie_title,
        'language': 'en-US'
    }
    
    try:
        response = requests.get(SEARCH_URL, params=params)
        response.raise_for_status() 
        data = response.json()
        
        if data['results']:
       
            first_result = data['results'][0]
            poster_path = first_result.get('poster_path')
            
            if poster_path:
                full_poster_url = f"{IMAGE_BASE_URL}{poster_path}"
                return full_poster_url
            else:
                return None 
        else:
            return None 
            
    except requests.exceptions.RequestException as e:
      
        print(f"Error API untuk '{movie_title}': {e}")
        return None

def add_posters_to_csv(input_csv_path="disney_movies.csv", output_csv_path="disney_movies_with_posters.csv"):
    """Fungsi utama untuk membaca, memproses, dan menyimpan data film."""
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: File CSV tidak ditemukan di {input_csv_path}")
        return
        
    print(f"Memulai pemrosesan {len(df)} film dari '{input_csv_path}'...")
    
    
    if 'poster_url' not in df.columns:
        df['poster_url'] = None
        
  
    for index, row in df.iterrows():
        title = row['movie_title']
        
      
        if pd.notna(row['poster_url']) and row['poster_url'] != '':
             print(f"[{index+1}/{len(df)}] Poster untuk '{title}' sudah ada. Melewati.")
             continue
        
        poster_url = get_poster_url(title)
        
        if poster_url:
          
            df.loc[index, 'poster_url'] = poster_url
            print(f"[{index+1}/{len(df)}] SUKSES: {title}")
        else:
            print(f"[{index+1}/{len(df)}] GAGAL: {title}. Poster tidak ditemukan.")
            
      
        time.sleep(0.2) 
        
    df.to_csv(output_csv_path, index=False)
    print("\n=======================================================")
    print(f"Proses Selesai. Data poster telah ditambahkan dan disimpan ke: {output_csv_path}")
    print("=======================================================")

if __name__ == "__main__":
    add_posters_to_csv()