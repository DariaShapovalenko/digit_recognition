import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageOps
import json
import os
from neural_net import NeuralNetwork, normalize


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Розпізнавання цифер")
        self.root.geometry("600x680")
        self.root.configure(bg="#f8e6f1")
        self.root.resizable(False, False)
        
        try:
            self.model = self.load_model("digit_model.json")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.root.destroy()
            return
        
        self.drawing = False
        self.last_x = 0
        self.last_y = 0
        self.line_width = 15
        
        self.create_widgets()
        self.create_canvas()
        self.create_drawing_image()
    
    def create_widgets(self):
       
        instructions = tk.Label(
            self.root,
            text="Намалюйте цифру (0-9) у полі нижче",
            font=("Helvetica", 12),
            bg="#f8e6f1",
            pady=5
        )
        instructions.pack()
        
        self.button_frame = tk.Frame(self.root, bg="#f8e6f1")
        self.button_frame.pack(pady=10)
        
        self.clear_button = tk.Button(
            self.button_frame,
            text="Очистити поле",
            font=("Helvetica", 12),
            command=self.clear_canvas,
            bg="#f7b2b7",
            fg="white",
            relief=tk.RAISED,
            padx=15, 
            pady=5
        )
        self.clear_button.grid(row=0, column=0, padx=10)
        
        self.classify_button = tk.Button(
            self.button_frame,
            text="Розпізнати",
            font=("Helvetica", 12),
            command=self.classify_drawing,
            bg="#f7b2b7",
            fg="white",
            relief=tk.RAISED,
            padx=15, 
            pady=5
        )
        self.classify_button.grid(row=0, column=1, padx=10)
        
        self.result_frame = tk.Frame(self.root, bg="#f8e6f1")
        self.result_frame.pack(pady=15)
        
        tk.Label(
            self.result_frame, 
            text="Прогноз:", 
            font=("Helvetica", 16),
            bg="#f8e6f1"
        ).pack(side=tk.LEFT, padx=5)
        
        self.prediction_label = tk.Label(
            self.result_frame, 
            text="--", 
            font=("Helvetica", 28, "bold"),
            bg="#f8e6f1", 
            fg="#333333",
            width=2
        )
        self.prediction_label.pack(side=tk.LEFT)
        
        self.confidence_frame = tk.Frame(self.root, bg="#f8e6f1")
       
        
        tk.Label(
            self.confidence_frame, 
            text="Ймовірність:", 
            font=("Helvetica", 14),
            bg="#f8e6f1"
        )
        
        self.confidence_label = tk.Label(
            self.confidence_frame, 
            text="--", 
            font=("Helvetica", 14),
            bg="#f8e6f1"
        )
        
    
    def create_canvas(self):
        self.canvas_frame = tk.Frame(
            self.root, 
            bg="#f8e6f1", 
            bd=2, 
            relief=tk.SUNKEN
        )
        self.canvas_frame.pack(pady=10)
        
        self.canvas_size = 280
        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black",
            cursor="circle"
        )
        self.canvas.pack()
        
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
    
    def create_drawing_image(self):
        self.drawing_image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.drawing_draw = ImageDraw.Draw(self.drawing_image)
    
    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def draw(self, event):
        if self.drawing:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=self.line_width, fill="white", capstyle=tk.ROUND, smooth=True
            )
            self.drawing_draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=255, width=self.line_width
            )
            self.last_x = event.x
            self.last_y = event.y
    
    def stop_drawing(self, event):
        self.drawing = False
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.create_drawing_image()
        self.prediction_label.config(text="--")
        self.confidence_label.config(text="--")
    
    def classify_drawing(self):
        if np.sum(np.array(self.drawing_image)) == 0:
            messagebox.showinfo("Info", "Будь ласка, спочатку намалюйте цифру!")
            return
        
        try:
            processed_image = self.preprocess_drawn_image()
            digit, confidence = self.predict_digit(processed_image)
            self.prediction_label.config(text=str(digit))
            self.confidence_label.config(text=f"{confidence:.2f}%")
        except Exception as e:
            messagebox.showerror("Error", f"Не вдалося розпізнати цифру: {e}")
    
    def preprocess_drawn_image(self):
        bbox = self.drawing_image.getbbox()
        
        if bbox is None:
            return np.zeros(784)
        
        cropped = self.drawing_image.crop(bbox)
        padded = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        
        crop_width, crop_height = cropped.size
        paste_x = (self.canvas_size - crop_width) // 2
        paste_y = (self.canvas_size - crop_height) // 2
        
        padded.paste(cropped, (paste_x, paste_y))
        resized = padded.resize((28, 28), Image.Resampling.LANCZOS)
        
        image_data = np.array(resized).reshape(-1).astype(np.float32)
        return normalize(image_data)
    
    def load_model(self, filename="digit_model.json"):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found!")
            
        with open(filename, "r") as f:
            model = json.load(f)
            
        nn = NeuralNetwork(model['layer_sizes'], model['activation'], model['learning_rate'])
        nn.weights = [np.array(w) for w in model['weights']]
        nn.biases = [np.array(b) for b in model['biases']]
        return nn
    
    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("L")
        self.display_loaded_image(image)
        image = ImageOps.invert(image)
        image = image.resize((28, 28))
        
        image_data = np.array(image).reshape(-1).astype(np.float32)
        return normalize(image_data)
    
    def display_loaded_image(self, image):
        self.clear_canvas()
        
        resized = image.resize((self.canvas_size, self.canvas_size))
        inverted = ImageOps.invert(resized)
        
        self.drawing_image = inverted
        self.drawing_draw = ImageDraw.Draw(self.drawing_image)
        
        photo = ImageTk.PhotoImage(inverted)
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.image = photo
    
    def predict_digit(self, image_data):
        probabilities = self.model.predict(image_data)
        predicted_class = np.argmax(probabilities)
        confidence = float(probabilities[predicted_class][0]) * 100
        return predicted_class, confidence
    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image", 
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
        )
        
        if file_path:
            try:
                image_data = self.preprocess_image(file_path)
                digit, confidence = self.predict_digit(image_data)
                self.prediction_label.config(text=str(digit))
                self.confidence_label.config(text=f"{confidence:.2f}%")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {e}")
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
