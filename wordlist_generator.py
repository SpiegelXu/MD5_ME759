import hashlib
import os

FILE_NAME = "my_wordlist.txt"
LINE_COUNT = 100_000_000  # 100M
TARGET_INDEX = 99_999_999 #choose the  last line


target_hash = ""
target_word = ""

with open(FILE_NAME, "w") as f:
    for i in range(LINE_COUNT):
        word = f"pass_{i}\n"
        f.write(word)
        
        #  Target word
        if i == TARGET_INDEX:
            target_word = word.strip()
            # calculate the outcome of chosen word's MD5
            target_hash = hashlib.md5(target_word.encode()).hexdigest()
        
        # print the process
        if i % 10_000_000 == 0:
            print(f"process: {i // 1_000_000}M / {LINE_COUNT // 1_000_000}M")

print("-" * 30)
print(f"Generating Finished!: {os.path.getsize(FILE_NAME) / (1024**2):.2f} MB")
print(f"Target Word: {target_word}")
print(f"Target Hash: {target_hash}")
print(f"Use the commande below to test: ")
print(f"./md5_openMP {FILE_NAME} {target_hash}")