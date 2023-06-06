#%% 
import tkinter as tk
import sys
import SuperResGuiCompatible as SuperRes
import threading

class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

def start():
    global stop_flag
    global threads
    stop_flag = False
    folder_path = entry.get()
    thread = threading.Thread(target=SuperRes.superresolution, args=(folder_path,lambda: stop_flag))
    thread.start()
    threads.append(thread)
    text.insert(tk.END, "The script has started with the folder path: " + folder_path + "\n")

def stop():
    global stop_flag
    global threads
    stop_flag = True
    for thread in threads:
        thread.join()
    root.destroy()

root = tk.Tk()
root.title("Python GUI")

label = tk.Label(root, text="Enter the file directory:")
label.pack()

entry = tk.Entry(root)
entry.pack()

start_button = tk.Button(root, text="Start", command=start)
start_button.pack()

stop_button = tk.Button(root, text="Stop", command=stop)
stop_button.pack()

text = tk.Text(root)
text.pack()

sys.stdout = StdoutRedirector(text)

stop_flag = False
threads = []

root.mainloop()
