import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import mysql.connector
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sklearn
import warnings
from PIL import Image, ImageTk
import csv

# Suppress warnings
warnings.filterwarnings("ignore")

# Database Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "student_performance_db",
}

# Initialize database
def init_db():
    try:
        conn = mysql.connector.connect(
            host=DB_CONFIG["host"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        cursor.execute(f"USE {DB_CONFIG['database']}")

        # Create users table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            role ENUM('admin', 'student') NOT NULL,
            first_login BOOLEAN DEFAULT TRUE
        )
        """
        )

        # Create student_data table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS student_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) NOT NULL,
            Hours_Studied FLOAT,
            Attendance FLOAT,
            Parental_Involvement ENUM('Low', 'Medium', 'High'),
            Access_to_Resources ENUM('Low', 'Medium', 'High'),
            Extracurricular_Activities ENUM('No', 'Yes'),
            Sleep_Hours FLOAT,
            Previous_Scores FLOAT,
            Motivation_Level ENUM('Low', 'Medium', 'High'),
            Internet_Access ENUM('No', 'Yes'),
            Tutoring_Sessions INT,
            Family_Income ENUM('Low', 'Medium', 'High'),
            Teacher_Quality ENUM('Low', 'Medium', 'High'),
            School_Type ENUM('Public', 'Private'),
            Peer_Influence ENUM('Negative', 'Neutral', 'Positive'),
            Physical_Activity INT,
            Learning_Disabilities ENUM('No', 'Yes'),
            Parental_Education_Level ENUM('High School', 'College', 'Postgraduate'),
            Distance_from_Home ENUM('Near', 'Moderate', 'Far'),
            Gender ENUM('Male', 'Female'),
            Exam_Score FLOAT,
            FOREIGN KEY (username) REFERENCES users(username)
        )
        """
        )

        # Create prediction_history table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            admin_username VARCHAR(50) NOT NULL,
            student_username VARCHAR(50) NOT NULL,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            predicted_score FLOAT NOT NULL,
            FOREIGN KEY (admin_username) REFERENCES users(username),
            FOREIGN KEY (student_username) REFERENCES users(username)
        )
        """
        )

        # Create admin if not exists
        cursor.execute(
            "INSERT IGNORE INTO users (username, password, role) VALUES ('admin', 'admin123', 'admin')"
        )

        conn.commit()
        cursor.close()
        conn.close()
        return True
    except mysql.connector.Error as err:
        messagebox.showerror("Database Error", f"Error initializing database: {err}")
        return False

class GradientButton(tk.Canvas):
    def __init__(self, master=None, text="", command=None, 
                 colors=("#4f46e5", "#6366f1"), width=300, height=40, 
                 font=("Segoe UI", 12), corner_radius=20, **kwargs):
        super().__init__(master, width=width, height=height, 
                        highlightthickness=0, bd=0, **kwargs)
        
        self.command = command
        self.colors = colors
        self.corner_radius = corner_radius
        self.font = font
        self.text = text
        self.width = width
        self.height = height
        
        self.bind("<Button-1>", self._on_click)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        
        self._draw_button()
    
    def _draw_button(self, hover=False):
        self.delete("all")
        
        # Create gradient
        if hover:
            colors = [self._lighten_color(c, 20) for c in self.colors]
        else:
            colors = self.colors
        
        for i in range(self.width):
            ratio = i / self.width
            r = int(colors[0][1:3], 16) * (1 - ratio) + int(colors[1][1:3], 16) * ratio
            g = int(colors[0][3:5], 16) * (1 - ratio) + int(colors[1][3:5], 16) * ratio
            b = int(colors[0][5:7], 16) * (1 - ratio) + int(colors[1][5:7], 16) * ratio
            color = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
            
            # Draw rounded rectangle parts
            if i < self.corner_radius or i > self.width - self.corner_radius:
                self.create_line(i, self.corner_radius, i, self.height - self.corner_radius, fill=color)
            else:
                self.create_line(i, 0, i, self.height, fill=color)
        
        # Draw rounded corners
        self._draw_rounded_corners(colors[0] if not hover else colors[0])
        
        # Add text
        self.create_text(
            self.width/2, self.height/2,
            text=self.text,
            fill="white",
            font=self.font,
            anchor="center"
        )
    
    def _draw_rounded_corners(self, color):
        # Top-left
        self.create_arc(
            0, 0, 
            self.corner_radius*2, self.corner_radius*2,
            start=90, extent=90,
            fill=color, outline=color
        )
        # Top-right
        self.create_arc(
            self.width-self.corner_radius*2, 0,
            self.width, self.corner_radius*2,
            start=0, extent=90,
            fill=color, outline=color
        )
        # Bottom-left
        self.create_arc(
            0, self.height-self.corner_radius*2,
            self.corner_radius*2, self.height,
            start=180, extent=90,
            fill=color, outline=color
        )
        # Bottom-right
        self.create_arc(
            self.width-self.corner_radius*2, self.height-self.corner_radius*2,
            self.width, self.height,
            start=270, extent=90,
            fill=color, outline=color
        )
    
    def _lighten_color(self, color, amount):
        r = min(255, int(color[1:3], 16) + amount)
        g = min(255, int(color[3:5], 16) + amount)
        b = min(255, int(color[5:7], 16) + amount)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _on_click(self, event):
        if self.command:
            self.command()
    
    def _on_enter(self, event):
        self._draw_button(hover=True)
    
    def _on_leave(self, event):
        self._draw_button(hover=False)

# PyTorch Neural Network Model
class PerformancePredictor(nn.Module):
    def __init__(self, input_size):
        super(PerformancePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Machine Learning Model Functions
def train_model():
    try:
        df = pd.read_csv("StudentPerformanceFactors.csv")

        # Preprocessing
        X = df.drop("Exam_Score", axis=1)
        y = df["Exam_Score"]

        # Identify categorical and numerical columns
        categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
        numerical_cols = [
            col for col in X.columns if X[col].dtype in ["int64", "float64"]
        ]

        # Preprocessor - handle different scikit-learn versions
        if sklearn.__version__ >= "1.2":
            # For newer scikit-learn versions (1.2+)
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numerical_cols),
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        categorical_cols,
                    ),
                ]
            )
        else:
            # For older scikit-learn versions (<1.2)
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numerical_cols),
                    (
                        "cat",
                        OneHotEncoder(handle_unknown="ignore", sparse=False),
                        categorical_cols,
                    ),
                ]
            )

        # Apply transformations
        X_processed = preprocessor.fit_transform(X)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_processed, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Initialize model
        model = PerformancePredictor(X_tensor.shape[1])

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train model
        num_epochs = 100
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Save model and preprocessor
        torch.save(model.state_dict(), "student_performance_model.pth")
        joblib.dump(preprocessor, "preprocessor.joblib")

        return True
    except Exception as e:
        messagebox.showerror("Training Error", f"Model training failed: {str(e)}")
        return False

def predict_performance(student_data):
    try:
        if not (
            os.path.exists("student_performance_model.pth")
            and os.path.exists("preprocessor.joblib")
        ):
            if not train_model():
                return None

        # Load preprocessor
        preprocessor = joblib.load("preprocessor.joblib")

        # Convert student data to DataFrame
        df = pd.DataFrame([student_data])

        # Preprocess data
        processed_data = preprocessor.transform(df)

        # Convert to tensor
        input_tensor = torch.tensor(processed_data, dtype=torch.float32)

        # Load model
        model = PerformancePredictor(input_tensor.shape[1])
        model.load_state_dict(torch.load("student_performance_model.pth"))
        model.eval()

        # Make prediction
        with torch.no_grad():
            prediction = model(input_tensor).item()
        return round(prediction, 2)
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Prediction failed: {str(e)}")
        return None

# GUI Application
class StudentPerformanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Performance Prediction System")
        self.root.geometry("1280x780")
        self.current_user = None
        self.current_role = None
        self.active_button = None  # Track the currently active button
        self.button_widgets = {}  # Store button references

        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(
            ".", background="#1a1a2e", foreground="#ffffff"  # Dark blue background
        )
        self.style.configure("TFrame", background="#1a1a2e")
        self.style.configure(
            "Card.TFrame",
            background="#2a2a4a",  # Card background
            borderwidth=0,
            relief=tk.FLAT,
        )
        self.style.configure(
            "Sidebar.TFrame",
            background="#16213e",
            width=500,  # Set default width here
            borderwidth=0,
            relief=tk.FLAT
        )
        self.style.configure("TLabel", background="#1a1a2e", foreground="#ffffff")
        self.style.configure("Header.TFrame", background="#0f3460")  # Darker header
        self.style.configure(
            "Header.TLabel",
            font=("Segoe UI", 16, "bold"),
            background="#0f3460",
            foreground="#ffffff",
            padding=10,
        )
        self.style.configure("Card.TLabel", background="#2a2a4a", foreground="#ffffff")
        self.style.configure("Sidebar.TFrame", background="#16213e")  # Sidebar
        self.style.configure(
            "TButton",
            font=("Segoe UI", 10),
            padding=6,
            background="#16213e",  # Sidebar color
            foreground="#ffffff",
            borderwidth=0,
            radius=15
        )
        self.style.map(
            "TButton",
            background=[("active", "#e94560"), ("pressed", "#e94560")],
            foreground=[("active", "#ffffff"), ("pressed", "#ffffff")],
        )
        self.style.configure(
            "Active.TButton",
            background="#e94560",  # Active button color
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
        )
        self.style.configure(
            "Accent.TButton",
            background="#e94560",  # Accent color
            foreground="#ffffff",
            font=("Segoe UI", 10, "bold"),
            borderradius=15,
        )
        self.style.map(
            "Accent.TButton", background=[("active", "#ff577f"), ("pressed", "#d13b5f")]
        )
        self.style.configure(
            "Treeview",
            background="#2a2a4a",  # Dark background
            foreground="#ffffff",  # White text
            fieldbackground="#2a2a4a",  # Background of fields
            rowheight=25,
        )  # Adjust row height as needed
        self.style.configure(
            "Treeview.Heading",
            background="#0f3460",  # Header background
            foreground="#ffffff",  # Header text color
            font=("Segoe UI", 10, "bold"),
        )
        self.style.map(
            "Treeview",
            background=[("selected", "#e94560")],  # Selection color
            foreground=[("selected", "#ffffff")],
        )  # Selected text color
        
        # Initialize database
        if not init_db():
            self.root.destroy()
            return

        self.configure_styles()
        self.show_login_screen()

    def configure_styles(self):
        self.style.theme_use('clam')
        
        # Modern color scheme
        self.bg_color = "#f8fafc"       # Very light gray
        self.primary_color = "#4f46e5"   # Indigo
        self.secondary_color = "#818cf8" # Lighter indigo
        self.card_color = "#ffffff"      # White
        self.text_color = "#1e293b"      # Dark gray
        self.light_text = "#64748b"      # Gray
        self.sidebar_color = "#1e293b"   # Dark blue-gray
        
        self.style.configure(
            'Card.TFrame',
            background=self.card_color,
            relief=tk.FLAT,
            borderwidth=0
        )
        
        self.style.configure(
            'Header.TLabel',
            font=('Segoe UI', 16, 'bold'),
            background=self.sidebar_color,
            foreground='white'
        )
        
        # Modern entry fields
        self.style.configure(
            'Modern.TEntry',
            fieldbackground=self.card_color,
            foreground=self.text_color,
            bordercolor="#e2e8f0",
            lightcolor="#e2e8f0",
            darkcolor="#e2e8f0",
            padding=10,
            width=500,
            height=10
        )

    def set_active_button(self, button_name, command=None):
        """Set the active state for the specified button and reset others"""
        if self.active_button:
            # Reset previous active button style
            self.button_widgets[self.active_button].configure(style="TButton")
        
        # Set new active button
        self.active_button = button_name
        self.button_widgets[button_name].configure(style="Active.TButton")
        
        # Execute the associated command if provided
        if command:
            command()

    def show_login_screen(self):
        """Display the modern login form as the initial screen"""
        self.clear_window()
        
        # Main container
        self.main_frame = tk.Frame(self.root, bg=self.bg_color)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Image/Illustration
        self.left_frame = tk.Frame(self.main_frame, bg=self.primary_color)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        
        try:
            img = Image.open("POLAC.png")
            img = img.resize((400, 400), Image.LANCZOS)
            self.login_img = ImageTk.PhotoImage(img)
            img_label = tk.Label(self.left_frame, image=self.login_img, bg=self.primary_color)
            img_label.pack(pady=100, padx=20)
        except:
            tk.Label(
                self.left_frame, 
                text="Student\nPerformance\nPrediction", 
                font=("Segoe UI", 32, "bold"), 
                fg="white", 
                bg=self.primary_color,
                justify=tk.CENTER
            ).pack(pady=200, padx=50)
        
        # Right side - Login Form
        self.right_frame = tk.Frame(self.main_frame, bg=self.card_color,highlightbackground="#000000", highlightthickness=1)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=50, pady=50)
        
        # Form header
        tk.Label(
            self.right_frame, 
            text="Welcome Back", 
            font=("Segoe UI", 28, "bold"), 
            fg=self.text_color, 
            bg=self.card_color
        ).pack(pady=(80, 10),padx=(150,0), anchor="w")
        
        tk.Label(
            self.right_frame, 
            text="Enter your credentials to access the system", 
            font=("Segoe UI", 10), 
            fg=self.light_text, 
            bg=self.card_color
        ).pack(anchor="w", pady=(0, 20),padx=(150,0))
        
        # Username field
        tk.Label(
            self.right_frame, 
            text="Username", 
            font=("Segoe UI", 10, "bold"), 
            fg=self.text_color, 
            bg=self.card_color
        ).pack(anchor="w", pady=(10, 5),padx=(100,0))
        
        self.username_entry = ttk.Entry(
            self.right_frame, 
            font=("Segoe UI", 11), 
            width=40,
            style='Modern.TEntry'
        )
        self.username_entry.pack(anchor="w", pady=(0, 20), ipady=1,padx=(100,100))
        
        # Password field
        tk.Label(
            self.right_frame, 
            text="Password", 
            font=("Segoe UI", 10, "bold"), 
            fg=self.text_color, 
            bg=self.card_color
        ).pack(anchor="w", pady=(10, 5),padx=(100,0))
        
        self.password_entry = ttk.Entry(
            self.right_frame, 
            font=("Segoe UI", 11),
             width=40,
            show="•",
            style='Modern.TEntry'
        )
        self.password_entry.pack(anchor="w", pady=(0, 10), ipady=1,padx=(100,100))
        
        # Login button
        self.login_btn = GradientButton(
            self.right_frame,
            text="Login",
            command=self.login,
            colors=(self.primary_color, self.secondary_color),
            width=250,
            height=45,
            corner_radius=12,
            bg=self.card_color
        )
        self.login_btn.pack(pady=(30, 50))
        
        # Bind Enter key to login
        self.root.bind('<Return>', lambda event: self.login())
        self.right_frame.place(relx=0.6, rely=0.5, anchor=tk.CENTER)

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Please enter both username and password")
            return
        
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT password, role, first_login FROM users WHERE username = %s",
                (username,),
            )
            result = cursor.fetchone()

            if result:
                stored_password, role, first_login = result

                if password == stored_password:
                    self.current_user = username
                    self.current_role = role

                    if role == "student" and first_login:
                        self.change_password(first_time=True)
                    else:
                        self.show_dashboard()
                else:
                    messagebox.showerror("Error", "Invalid credentials")
            else:
                messagebox.showerror("Error", "User not found")

            cursor.close()
            conn.close()
        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", f"Error during login: {err}")

    def change_password(self, first_time=False):
        self.clear_window()

        change_pwd_frame = ttk.Frame(self.root, style="TFrame")
        change_pwd_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header = ttk.Frame(change_pwd_frame, style="Header.TFrame")
        header.pack(fill=tk.X)
        ttk.Label(header, text="Change Password", style="Header.TLabel").pack(
            padx=20, pady=15
        )

        # Form
        form_container = ttk.Frame(change_pwd_frame)
        form_container.pack(pady=50)

        ttk.Label(form_container, text="New Password", font=("Segoe UI", 11)).grid(
            row=0, column=0, padx=10, pady=10, sticky="w"
        )
        self.new_pwd_entry = ttk.Entry(
            form_container, font=("Segoe UI", 11), width=30, show="*",foreground="black"
        )
        self.new_pwd_entry.grid(row=0, column=1, padx=10, pady=10)

        ttk.Label(form_container, text="Confirm Password", font=("Segoe UI", 11)).grid(
            row=1, column=0, padx=10, pady=10, sticky="w"
        )
        self.confirm_pwd_entry = ttk.Entry(
            form_container, font=("Segoe UI", 11), width=30, show="*",foreground="black"
        )
        self.confirm_pwd_entry.grid(row=1, column=1, padx=10, pady=10)

        btn_frame = ttk.Frame(form_container)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=20)

        ttk.Button(
            btn_frame,
            text="Change Password",
            command=lambda: self.save_password(first_time),
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=10)

        # Center the form
        form_container.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def save_password(self, first_time):
        new_pwd = self.new_pwd_entry.get()
        confirm_pwd = self.confirm_pwd_entry.get()

        if not new_pwd or not confirm_pwd:
            messagebox.showerror("Error", "Please enter both fields")
            return

        if new_pwd != confirm_pwd:
            messagebox.showerror("Error", "Passwords do not match")
            return

        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Update password and reset first_login flag
            cursor.execute(
                "UPDATE users SET password = %s, first_login = FALSE WHERE username = %s",
                (new_pwd, self.current_user),
            )
            conn.commit()

            cursor.close()
            conn.close()

            messagebox.showinfo("Success", "Password changed successfully")

            if first_time:
                self.show_dashboard()
            else:
                self.show_profile()
        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", f"Error changing password: {err}")

    def show_dashboard(self):
        self.clear_window()

        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Sidebar
        sidebar = ttk.Frame(main_container, style="Sidebar.TFrame", width=400)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, ipadx=35)

        try:
            icon_img = tk.PhotoImage(file="POLAC.png")
            icon_img = icon_img.subsample(7,7)
            icon_label = ttk.Label(sidebar, image=icon_img, background="#16213e")
            icon_label.image = icon_img  # Keep a reference to prevent garbage collection
            icon_label.pack(pady=(20, 10))
        except Exception as e:
            print(f"Could not load icon: {e}")

        # Sidebar buttons - store references in self.button_widgets
        self.button_widgets["home"] = ttk.Button(
            sidebar, 
            text="Home", 
            style="TButton", 
            command=lambda: self.set_active_button("home", self.show_home)
        )
        self.button_widgets["home"].pack(pady=0, padx=0, fill=tk.X)

        if self.current_role == "admin":
            self.button_widgets["student_data"] = ttk.Button(
                sidebar,
                text="Student Data",
                style="TButton",
                command=lambda: self.set_active_button("student_data", self.show_student_data_admin),
            )
            self.button_widgets["student_data"].pack(pady=0, padx=0, fill=tk.X)
            
            self.button_widgets["prediction"] = ttk.Button(
                sidebar,
                text="Prediction",
                style="TButton",
                command=lambda: self.set_active_button("prediction", self.show_prediction),
            )
            self.button_widgets["prediction"].pack(pady=0, padx=0, fill=tk.X)
            
            self.button_widgets["batch_predict"] = ttk.Button(
                sidebar,
                text="Batch Predict",
                style="TButton",
                command=lambda: self.set_active_button("batch_predict", self.show_batch_predict),
            )
            self.button_widgets["batch_predict"].pack(pady=0, padx=0, fill=tk.X)
            
            self.button_widgets["analysis"] = ttk.Button(
                sidebar,
                text="Analysis",
                style="TButton",
                command=lambda: self.set_active_button("analysis", self.show_analysis),
            )
            self.button_widgets["analysis"].pack(pady=0, padx=0, fill=tk.X)
        else:
            self.button_widgets["student_data"] = ttk.Button(
                sidebar,
                text="Student Data",
                style="TButton",
                command=lambda: self.set_active_button("student_data", self.show_student_data),
            )
            self.button_widgets["student_data"].pack(pady=0, padx=0, fill=tk.X)

        self.button_widgets["profile"] = ttk.Button(
            sidebar, 
            text="Profile", 
            style="TButton", 
            command=lambda: self.set_active_button("profile", self.show_profile)
        )
        self.button_widgets["profile"].pack(pady=0, padx=0, fill=tk.X)

        ttk.Button(
            sidebar, text="Logout", style="Accent.TButton", command=self.logout
        ).pack(pady=0, padx=0, fill=tk.X, side=tk.BOTTOM)

        # Content area
        self.content_area = ttk.Frame(main_container)
        self.content_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Set home as active by default
        self.set_active_button("home", self.show_home)

    def show_home(self):
        self.clear_content_area()

        home_frame = ttk.Frame(self.content_area)
        home_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(
            home_frame,
            text="Welcome to Student Performance Prediction System",
            font=("Segoe UI", 18, "bold"),
        ).pack(pady=20)

        # Introduction text
        intro_text = """
        This application provides a comprehensive solution for predicting student academic performance 
        based on various factors such as study habits, attendance, family background, and more.
        
        Key Features:
        - Role-based access system (Admin and Student)
        - Student performance prediction using deep learning
        - Data analysis and visualization
        - Student profile management
        
        For students:
        - Enter and update your academic data
        - View your performance history
        
        For administrators:
        - Create and manage student profiles
        - Predict student performance
        - Analyze historical data and trends
        """

        intro_label = ttk.Label(
            home_frame, text=intro_text, font=("Segoe UI", 11), justify=tk.LEFT
        )
        intro_label.pack(pady=20, padx=20, fill=tk.X)

        # Statistics frame
        stats_frame = ttk.Frame(home_frame)
        stats_frame.pack(fill=tk.X, pady=20)

        # Get stats from database
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Total students
            cursor.execute("SELECT COUNT(*) FROM users WHERE role='student'")
            total_students = cursor.fetchone()[0]

            # Data entries
            cursor.execute("SELECT COUNT(*) FROM student_data")
            data_entries = cursor.fetchone()[0]

            # Predictions made
            cursor.execute("SELECT COUNT(*) FROM prediction_history")
            predictions = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            # Display stats
            ttk.Label(
                stats_frame,
                text=f"Total Students\n{total_students}",
                font=("Segoe UI", 12),
                relief=tk.RIDGE,
                padding=10,
            ).pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

            ttk.Label(
                stats_frame,
                text=f"Data Entries\n{data_entries}",
                font=("Segoe UI", 12),
                relief=tk.RIDGE,
                padding=10,
            ).pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

            ttk.Label(
                stats_frame,
                text=f"Predictions\n{predictions}",
                font=("Segoe UI", 12),
                relief=tk.RIDGE,
                padding=10,
            ).pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", f"Could not fetch stats: {err}")

    def show_profile(self):
        self.clear_content_area()

        profile_frame = ttk.Frame(self.content_area)
        profile_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(
            profile_frame, text="User Profile", font=("Segoe UI", 16, "bold")
        ).pack(pady=10, anchor=tk.W)

        # Profile info
        info_frame = ttk.LabelFrame(profile_frame, text="Account Information")
        info_frame.pack(fill=tk.X, pady=10, padx=10)

        ttk.Label(
            info_frame, text=f"Username: {self.current_user}", font=("Segoe UI", 11)
        ).pack(anchor=tk.W, padx=10, pady=5)
        ttk.Label(
            info_frame, text=f"Role: {self.current_role}", font=("Segoe UI", 11)
        ).pack(anchor=tk.W, padx=10, pady=5)

        # Change password button
        ttk.Button(
            profile_frame, text="Change Password", command=self.change_password
        ).pack(pady=10, anchor=tk.W)

        # Student data section (for students)
        if self.current_role == "student":
            ttk.Label(
                profile_frame, text="Your Data", font=("Segoe UI", 14, "bold")
            ).pack(pady=10, anchor=tk.W)
            self.display_student_data(profile_frame, self.current_user)

    def display_student_data(self, parent_frame, username):
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM student_data WHERE username = %s", (username,)
            )
            data = cursor.fetchone()
            cursor.close()
            conn.close()

            if not data:
                ttk.Label(
                    parent_frame, text="No data available", font=("Segoe UI", 11)
                ).pack(anchor=tk.W, padx=10, pady=5)
                return

            # Create a frame for data display
            data_frame = ttk.Frame(parent_frame)
            data_frame.pack(fill=tk.X, pady=10, padx=10)

            # Display data in a grid
            row = 0
            for key, value in data.items():
                if key in ["id", "username"]:
                    continue

                ttk.Label(
                    data_frame,
                    text=f"{key.replace('_', ' ')}:",
                    font=("Segoe UI", 11, "bold"),
                ).grid(row=row, column=0, padx=5, pady=2, sticky="e")
                ttk.Label(data_frame, text=str(value), font=("Segoe UI", 11)).grid(
                    row=row, column=1, padx=5, pady=2, sticky="w"
                )
                row += 1

        except mysql.connector.Error as err:
            messagebox.showerror(
                "Database Error", f"Could not fetch student data: {err}"
            )

    def show_student_data(self):
        self.clear_content_area()

        data_frame = ttk.Frame(self.content_area)
        data_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(
            data_frame, text="Student Data Entry", font=("Segoe UI", 16, "bold")
        ).pack(pady=10, anchor=tk.W)

        # Check if data already exists
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM student_data WHERE username = %s",
                (self.current_user,),
            )
            data_exists = cursor.fetchone()[0] > 0
            cursor.close()
            conn.close()
        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", f"Error checking data: {err}")
            return

        # Form for data entry
        form_frame = ttk.LabelFrame(data_frame, text="Enter Your Data")
        form_frame.pack(fill=tk.X, pady=10, padx=10)

        # Form fields
        fields = [
            "Hours_Studied",
            "Attendance",
            "Sleep_Hours",
            "Previous_Scores",
            "Tutoring_Sessions",
            "Physical_Activity",
        ]

        categorical_fields = {
            "Parental_Involvement": ["Low", "Medium", "High"],
            "Access_to_Resources": ["Low", "Medium", "High"],
            "Extracurricular_Activities": ["No", "Yes"],
            "Motivation_Level": ["Low", "Medium", "High"],
            "Internet_Access": ["No", "Yes"],
            "Family_Income": ["Low", "Medium", "High"],
            "Teacher_Quality": ["Low", "Medium", "High"],
            "School_Type": ["Public", "Private"],
            "Peer_Influence": ["Negative", "Neutral", "Positive"],
            "Learning_Disabilities": ["No", "Yes"],
            "Parental_Education_Level": ["High School", "College", "Postgraduate"],
            "Distance_from_Home": ["Near", "Moderate", "Far"],
            "Gender": ["Male", "Female"],
        }

        self.data_entries = {}

        # Create numerical fields
        row = 0
        for field in fields:
            ttk.Label(form_frame, text=f"{field.replace('_', ' ')}:").grid(
                row=row, column=0, padx=5, pady=5, sticky="e"
            )
            entry = ttk.Entry(form_frame, foreground="black", width=10)
            entry.grid(row=row, column=1, padx=5, pady=5, sticky="w")
            self.data_entries[field] = entry
            row += 1

        # Create categorical fields
        for field, options in categorical_fields.items():
            ttk.Label(form_frame, text=f"{field.replace('_', ' ')}:").grid(
                row=row, column=0, padx=5, pady=5, sticky="e"
            )
            var = tk.StringVar()
            combobox = ttk.Combobox(
                form_frame,
                textvariable=var,
                values=options,
                state="readonly",
                foreground="black",width=20
            )
            combobox.grid(row=row, column=1, padx=5, pady=5, sticky="w")
            self.data_entries[field] = var
            row += 1

        # Buttons
        btn_frame = ttk.Frame(data_frame)
        btn_frame.pack(pady=10)

        ttk.Button(
            btn_frame,
            text="Submit Data",
            command=lambda: self.save_student_data(data_exists),
        ).pack(side=tk.LEFT, padx=5)

    def save_student_data(self, data_exists):
        data = {}

        # Get values from form
        for key, widget in self.data_entries.items():
            if isinstance(widget, ttk.Entry):
                value = widget.get()
                # Validate numerical fields
                if not value.replace(".", "", 1).isdigit():
                    messagebox.showerror(
                        "Error", f"Invalid value for {key.replace('_', ' ')}"
                    )
                    return
                data[key] = float(value) if "." in value else int(value)
            else:
                data[key] = widget.get()
                if not data[key]:
                    messagebox.showerror(
                        "Error", f"Please select a value for {key.replace('_', ' ')}"
                    )
                    return

        # Add username
        data["username"] = self.current_user

        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()

            if data_exists:
                # Update existing data
                set_clause = ", ".join(
                    [f"{key} = %s" for key in data.keys() if key != "username"]
                )
                query = f"UPDATE student_data SET {set_clause} WHERE username = %s"
                values = [val for key, val in data.items() if key != "username"] + [
                    self.current_user
                ]
            else:
                # Insert new data
                columns = ", ".join(data.keys())
                placeholders = ", ".join(["%s"] * len(data))
                query = f"INSERT INTO student_data ({columns}) VALUES ({placeholders})"
                values = list(data.values())

            cursor.execute(query, values)
            conn.commit()

            cursor.close()
            conn.close()

            messagebox.showinfo("Success", "Data saved successfully")
            self.show_student_data()
        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", f"Error saving data: {err}")

    def show_student_data_admin(self):
        self.clear_content_area()

        data_frame = ttk.Frame(self.content_area)
        data_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(
            data_frame, text="Student Management", font=("Segoe UI", 16, "bold")
        ).pack(pady=10, anchor=tk.W)

        # Create student button
        ttk.Button(
            data_frame, text="Create New Student", command=self.show_create_student
        ).pack(pady=10, anchor=tk.W)

        # Student list
        list_frame = ttk.LabelFrame(data_frame, text="Student List")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Treeview for student list
        columns = ("id", "username")
        tree = ttk.Treeview(
            list_frame, columns=columns, show="headings", style="Treeview"
        )
        tree.heading("id", text="ID")
        tree.heading("username", text="Username")
        tree.column("id", width=100, anchor=tk.CENTER)
        tree.column("username", width=300)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)

        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Fill student list
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute("SELECT id, username FROM users WHERE role='student'")

            for row in cursor.fetchall():
                tree.insert("", tk.END, values=row)

            cursor.close()
            conn.close()
        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", f"Error fetching students: {err}")

        # View data button
        ttk.Button(
            data_frame,
            text="View Student Data",
            command=lambda: self.view_student_data(tree),
        ).pack(pady=10, anchor=tk.W)

    def show_create_student(self):
        self.clear_content_area()

        create_frame = ttk.Frame(self.content_area)
        create_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(
            create_frame, text="Create New Student", font=("Segoe UI", 16, "bold")
        ).pack(pady=10, anchor=tk.W)

        # Notebook for single vs batch creation
        notebook = ttk.Notebook(create_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # Tab 1: Single Student Creation
        single_frame = ttk.Frame(notebook)
        notebook.add(single_frame, text="Single Student")

        # Form
        form_frame = ttk.Frame(single_frame)
        form_frame.pack(fill=tk.X, pady=10)

        ttk.Label(form_frame, text="Username:").grid(
            row=0, column=0, padx=5, pady=5, sticky="e"
        )
        self.new_username = ttk.Entry(form_frame, font=("Segoe UI", 11), foreground="black", width=30)
        self.new_username.grid(row=0, column=1, padx=5, pady=12, sticky="w")

        ttk.Label(form_frame, text="Roll Number:").grid(
            row=1, column=0, padx=5, pady=5, sticky="e"
        )
        self.new_rollno = ttk.Entry(form_frame, font=("Segoe UI", 11), foreground="black", width=30)
        self.new_rollno.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        # Buttons
        btn_frame = ttk.Frame(single_frame)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Create Student", command=self.create_student).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(btn_frame, text="Back", command=self.show_student_data_admin).pack(
            side=tk.LEFT, padx=5
        )

        # Tab 2: Batch Student Creation
        batch_frame = ttk.Frame(notebook)
        notebook.add(batch_frame, text="Batch Import")

        # CSV Import Section
        csv_frame = ttk.LabelFrame(batch_frame, text="Import Students from CSV")
        csv_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(csv_frame, text="CSV File Format:", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=5)
        ttk.Label(csv_frame, text="The CSV file should have one column named 'username' with student usernames.").pack(anchor=tk.W)
       

        # File selection
        file_frame = ttk.Frame(csv_frame)
        file_frame.pack(fill=tk.X, pady=10)

        ttk.Button(
            file_frame,
            text="Select CSV File",
            command=self.select_student_csv,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=5)

        self.csv_file_path_label = ttk.Label(file_frame, text="No file selected")
        self.csv_file_path_label.pack(side=tk.LEFT, padx=10)

        # Preview area
        preview_frame = ttk.LabelFrame(csv_frame, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create Treeview for preview
        self.csv_preview_tree = ttk.Treeview(preview_frame)
        self.csv_preview_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add scrollbars
        y_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.csv_preview_tree.yview)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.csv_preview_tree.configure(yscrollcommand=y_scrollbar.set)

        x_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.csv_preview_tree.xview)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.csv_preview_tree.configure(xscrollcommand=x_scrollbar.set)

        # Import button
        btn_frame2 = ttk.Frame(csv_frame)
        btn_frame2.pack(pady=10)

        self.import_btn = ttk.Button(
            btn_frame2,
            text="Import Students",
            command=self.import_students_from_csv,
            style="Accent.TButton",
            state=tk.DISABLED,
        )
        self.import_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(btn_frame2, text="Back", command=self.show_student_data_admin).pack(
            side=tk.LEFT, padx=5
        )

    def select_student_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.csv_file_path_label.config(text=file_path)
            self.current_student_csv = file_path
            self.preview_student_csv(file_path)
            self.import_btn.config(state=tk.NORMAL)

    def preview_student_csv(self, file_path):
        try:
            # Clear existing preview
            for item in self.csv_preview_tree.get_children():
                self.csv_preview_tree.delete(item)

            # Read CSV file
            with open(file_path, 'r') as file:
                reader = csv.DictReader(file)
                if 'username' not in reader.fieldnames:
                    messagebox.showerror("Error", "CSV file must have a 'username' column")
                    return

                # Set up Treeview columns
                self.csv_preview_tree["columns"] = reader.fieldnames
                self.csv_preview_tree.column("#0", width=0, stretch=tk.NO)

                for col in reader.fieldnames:
                    self.csv_preview_tree.heading(col, text=col)
                    self.csv_preview_tree.column(col, width=120)

                # Insert first 10 rows for preview
                for i, row in enumerate(reader):
                    if i >= 10:  # Limit to 10 rows for preview
                        break
                    self.csv_preview_tree.insert("", tk.END, values=[row[col] for col in reader.fieldnames])

        except Exception as e:
            messagebox.showerror("Error", f"Could not read CSV file: {str(e)}")

    def import_students_from_csv(self):
        if not hasattr(self, "current_student_csv"):
            messagebox.showerror("Error", "No CSV file selected")
            return

        try:
            # Read the entire CSV file
            with open(self.current_student_csv, 'r') as file:
                reader = csv.DictReader(file)
                if 'username' not in reader.fieldnames:
                    messagebox.showerror("Error", "CSV file must have a 'username' column")
                    return

                # Process each row
                total = 0
                success = 0
                errors = []

                conn = mysql.connector.connect(**DB_CONFIG)
                cursor = conn.cursor()

                for row in reader:
                    total += 1
                    username = row['username']
                    roll_number = row.get('roll_number', '')  # Optional field

                    try:
                        # Create student with default password
                        cursor.execute(
                            "INSERT INTO users (username, password, role) VALUES (%s, '12345678', 'student')",
                            (username,),
                        )
                        success += 1
                    except mysql.connector.Error as err:
                        errors.append(f"Username '{username}': {err}")

                conn.commit()
                cursor.close()
                conn.close()

                # Show results
                result_message = f"Processed {total} students\nSuccessfully created: {success}"
                if errors:
                    result_message += f"\n\nErrors ({len(errors)}):\n" + "\n".join(errors[:5])  # Show first 5 errors
                    if len(errors) > 5:
                        result_message += f"\n...and {len(errors)-5} more"

                messagebox.showinfo("Import Results", result_message)
                self.show_student_data_admin()

        except Exception as e:
            messagebox.showerror("Error", f"Error importing students: {str(e)}")

    def create_student(self):
        username = self.new_username.get()
        roll_number = self.new_rollno.get()

        if not username:
            messagebox.showerror("Error", "Please enter a username")
            return

        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Create student with default password
            cursor.execute(
                "INSERT INTO users (username, password, role) VALUES (%s, '12345678', 'student')",
                (username,),
            )
            conn.commit()

            cursor.close()
            conn.close()

            messagebox.showinfo(
                "Success",
                f"Student '{username}' created with default password '12345678'",
            )
            self.show_student_data_admin()
        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", f"Error creating student: {err}")

    def view_student_data(self, tree):
        selected = tree.focus()
        if not selected:
            messagebox.showerror("Error", "Please select a student")
            return

        student_id = tree.item(selected, "values")[0]
        username = tree.item(selected, "values")[1]

        self.clear_content_area()

        data_frame = ttk.Frame(self.content_area)
        data_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(
            data_frame, text=f"Student Data: {username}", font=("Segoe UI", 16, "bold")
        ).pack(pady=10, anchor=tk.W)

        # Display student data
        self.display_student_data(data_frame, username)

        # Back button
        ttk.Button(data_frame, text="Back", command=self.show_student_data_admin).pack(
            pady=10, anchor=tk.W
        )

    def show_prediction(self):
        self.clear_content_area()

        prediction_frame = ttk.Frame(self.content_area)
        prediction_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(
            prediction_frame,
            text="Student Performance Prediction",
            font=("Segoe UI", 16, "bold"),
        ).pack(pady=10, anchor=tk.W)

        # Student selection
        student_frame = ttk.Frame(prediction_frame)
        student_frame.pack(fill=tk.X, pady=10)

        ttk.Label(student_frame, text="Select Student:").pack(side=tk.LEFT, padx=5)

        self.student_var = tk.StringVar()
        student_cb = ttk.Combobox(
            student_frame, textvariable=self.student_var, state="readonly"
        )
        student_cb.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Load students
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute("SELECT username FROM users WHERE role='student'")
            students = [row[0] for row in cursor.fetchall()]
            student_cb["values"] = students
            cursor.close()
            conn.close()
        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", f"Error fetching students: {err}")

        # Prediction button
        ttk.Button(
            prediction_frame,
            text="Predict Performance",
            command=self.predict_student_performance,
        ).pack(pady=10, anchor=tk.W)

        # Result display
        self.prediction_result = ttk.Label(
            prediction_frame, text="", font=("Segoe UI", 14)
        )
        self.prediction_result.pack(pady=20)

        # History button
        ttk.Button(
            prediction_frame,
            text="View Prediction History",
            command=self.show_analysis,
            style="Accent.TButton",
        ).pack(pady=10, anchor=tk.W)

    def predict_student_performance(self):
        student_username = self.student_var.get()

        if not student_username:
            messagebox.showerror("Error", "Please select a student")
            return

        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM student_data WHERE username = %s", (student_username,)
            )
            data = cursor.fetchone()
            cursor.close()
            conn.close()

            if not data:
                messagebox.showerror("Error", "No data available for selected student")
                return

            # Remove non-feature columns
            data.pop("id", None)
            data.pop("username", None)
            data.pop("Exam_Score", None)

            # Predict performance
            prediction = predict_performance(data)

            if prediction is not None:
                # Create a frame for the prediction result and button
                result_frame = ttk.Frame(self.prediction_result.master)
                result_frame.pack(pady=20)
                
                # Display prediction
                ttk.Label(
                    result_frame, 
                    text=f"Predicted Exam Score: {prediction}",
                    font=("Segoe UI", 14)
                ).pack(side=tk.LEFT, padx=10)
                
                # Add heatmap button
                ttk.Button(
                    result_frame,
                    text="Show Factors Analysis",
                    command=lambda: self.show_student_heatmap(data),
                    style="Accent.TButton"
                ).pack(side=tk.LEFT, padx=10)

                # Save to history
                try:
                    conn = mysql.connector.connect(**DB_CONFIG)
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO prediction_history (admin_username, student_username, predicted_score) VALUES (%s, %s, %s)",
                        (self.current_user, student_username, prediction),
                    )
                    conn.commit()
                    cursor.close()
                    conn.close()
                except mysql.connector.Error as err:
                    messagebox.showerror(
                        "Database Error", f"Error saving prediction: {err}"
                    )
            else:
                self.prediction_result.config(text="Prediction failed")

        except mysql.connector.Error as err:
            messagebox.showerror(
                "Database Error", f"Error fetching student data: {err}"
            )

    def show_student_heatmap(self, student_data):
        """Show a heatmap of factors affecting the student's exam score"""
        # Create a new window
        heatmap_window = tk.Toplevel(self.root)
        heatmap_window.title("Factors Affecting Exam Score")
        heatmap_window.geometry("800x600")
        
        try:
            # Convert student data to DataFrame
            df = pd.DataFrame([student_data])
            
            # Remove non-feature columns
            df = df.drop(columns=['id', 'username', 'Exam_Score'], errors='ignore')
            
            # Calculate correlation with predicted score
            predicted_score = predict_performance(student_data)
            if predicted_score is None:
                raise ValueError("Could not generate prediction")
                
            # Add predicted score to dataframe for correlation calculation
            df['Predicted_Score'] = predicted_score
            
            # Convert categorical features to numerical for correlation
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df[col] = pd.factorize(df[col])[0]
                
            # Calculate correlations
            correlations = df.corr()['Predicted_Score'].drop('Predicted_Score')
            
            # Sort by absolute value of correlation
            correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
            
            # Create figure
            fig = plt.Figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            
            # Create bar plot of correlations
            sns.barplot(x=correlations.values, y=correlations.index, ax=ax, palette="coolwarm")
            
            # Customize plot
            ax.set_title("Factors Affecting Exam Score (Correlation)")
            ax.set_xlabel("Correlation Coefficient")
            ax.set_ylabel("Factor")
            ax.set_xlim(-1, 1)
            
            # Add value labels
            for i, v in enumerate(correlations.values):
                ax.text(v, i, f"{v:.2f}", color='black', ha='left' if v < 0 else 'right', va='center')
            
            # Embed in Tkinter
            canvas = FigureCanvasTkAgg(fig, master=heatmap_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add close button
            ttk.Button(
                heatmap_window, 
                text="Close", 
                command=heatmap_window.destroy,
                style="Accent.TButton"
            ).pack(pady=10)
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate heatmap: {str(e)}")
            heatmap_window.destroy()

    def show_batch_predict(self):
        self.clear_content_area()

        batch_frame = ttk.Frame(self.content_area)
        batch_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(
            batch_frame, text="Batch Prediction", font=("Segoe UI", 16, "bold")
        ).pack(pady=10, anchor=tk.W)

        # File selection
        file_frame = ttk.Frame(batch_frame)
        file_frame.pack(fill=tk.X, pady=10)

        ttk.Button(
            file_frame,
            text="Select CSV File",
            command=self.select_batch_file,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=5)

        self.file_path_label = ttk.Label(file_frame, text="No file selected")
        self.file_path_label.pack(side=tk.LEFT, padx=10)

        # Preview area
        preview_frame = ttk.LabelFrame(batch_frame, text="Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        preview_frame.config(height=300)  # Set minimum height

        # Create Treeview for preview
        self.preview_tree = ttk.Treeview(preview_frame)
        self.preview_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add scrollbar
        scrollbar = ttk.Scrollbar(
            preview_frame, orient=tk.VERTICAL, command=self.preview_tree.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.preview_tree.configure(yscrollcommand=scrollbar.set)
        # Add after vertical scrollbar setup
        x_scrollbar = ttk.Scrollbar(
            preview_frame, orient=tk.HORIZONTAL, command=self.preview_tree.xview
        )
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.preview_tree.configure(xscrollcommand=x_scrollbar.set)

        # Process button
        btn_frame = ttk.Frame(batch_frame)
        btn_frame.pack(pady=10)

        self.process_btn = ttk.Button(
            btn_frame,
            text="Process Predictions",
            command=self.process_batch_predictions,
            style="Accent.TButton",
            state=tk.DISABLED,
        )
        self.process_btn.pack(side=tk.LEFT, padx=5)

        self.graph_btn = ttk.Button(
            btn_frame,
            text="Graph",
            command=self.show_batch_graphs,
            style="Accent.TButton",
            state=tk.DISABLED,
        )
        self.graph_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = ttk.Button(
            btn_frame,
            text="Save Results",
            command=self.save_batch_results,
            style="Accent.TButton",
            state=tk.DISABLED,
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

    def select_batch_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.file_path_label.config(text=file_path)
            self.current_batch_file = file_path
            self.preview_file(file_path)
            self.process_btn.config(state=tk.NORMAL)

    def preview_file(self, file_path):
        try:
            # Clear existing preview
            for item in self.preview_tree.get_children():
                self.preview_tree.delete(item)

            # Read first 10 rows
            df = pd.read_csv(file_path)

            self.preview_tree["columns"] = list(df.columns)

            # Configure the special first column
            self.preview_tree.column(
                "#0", width=0, stretch=tk.NO
            )  # HIDE the first column

            for col in df.columns:
                self.preview_tree.heading(col, text=col)
                # Set width based on header length
                col_width = min(150, max(70, len(col) * 8 + 20))
                self.preview_tree.column(col, width=col_width, minwidth=50)

            # Insert data
            for _, row in df.iterrows():
                self.preview_tree.insert("", tk.END, values=tuple(row))

        except Exception as e:
            messagebox.showerror("Error", f"Could not read file: {str(e)}")

    def process_batch_predictions(self):
        if not hasattr(self, "current_batch_file"):
            messagebox.showerror("Error", "No file selected")
            return

        try:
            # Read the entire file
            self.batch_df = pd.read_csv(self.current_batch_file)

            # Validate columns
            required_columns = [
                "Hours_Studied",
                "Attendance",
                "Parental_Involvement",
                "Access_to_Resources",
                "Extracurricular_Activities",
                "Sleep_Hours",
                "Previous_Scores",
                "Motivation_Level",
                "Internet_Access",
                "Tutoring_Sessions",
                "Family_Income",
                "Teacher_Quality",
                "School_Type",
                "Peer_Influence",
                "Physical_Activity",
                "Learning_Disabilities",
                "Parental_Education_Level",
                "Distance_from_Home",
                "Gender",
            ]

            missing = [
                col for col in required_columns if col not in self.batch_df.columns
            ]
            if missing:
                messagebox.showerror(
                    "Error", f"Missing columns in CSV: {', '.join(missing)}"
                )
                return

            # Add prediction column
            self.batch_df["Predicted_Score"] = np.nan

            # Process each row
            total_rows = len(self.batch_df)
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Processing")
            progress_window.geometry("300x100")

            progress_label = ttk.Label(
                progress_window, text="Processing predictions...", font=("Segoe UI", 10)
            )
            progress_label.pack(pady=10)

            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(
                progress_window, variable=progress_var, maximum=total_rows
            )
            progress_bar.pack(fill=tk.X, padx=20, pady=5)

            progress_window.update()

            # Process predictions
            for i, row in self.batch_df.iterrows():
                student_data = row[required_columns].to_dict()
                prediction = predict_performance(student_data)
                self.batch_df.at[i, "Predicted_Score"] = prediction

                # Update progress
                progress_var.set(i + 1)
                progress_label.config(
                    text=f"Processing {i+1}/{total_rows} predictions..."
                )
                progress_window.update()

            progress_window.destroy()

            # Update preview with results
            self.update_preview_with_results()
            self.save_btn.config(state=tk.NORMAL)
            self.graph_btn.config(state=tk.NORMAL)
            messagebox.showinfo(
                "Success", f"Successfully processed {total_rows} predictions"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Batch prediction failed: {str(e)}")

    def update_preview_with_results(self):
        # Clear existing preview
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)

        # Get first 10 rows with predictions
        preview_df = self.batch_df

        # Set up Treeview columns
        self.preview_tree["columns"] = list(preview_df.columns)

        # Configure the special first column
        self.preview_tree.column("#0", width=0, stretch=tk.NO)  # HIDE the first column

        for col in preview_df.columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=100)

        # Insert data
        for _, row in preview_df.iterrows():
            self.preview_tree.insert("", tk.END, values=tuple(row))

        # Auto-size columns after inserting data
        self.autosize_columns()

    def autosize_columns(self):
        for col in self.preview_tree["columns"]:
            max_width = 80  # Minimum width
            # Find the maximum width needed for this column
            for item in self.preview_tree.get_children():
                value = self.preview_tree.set(item, col)
                width = len(str(value)) * 8 + 20  # Approximate pixel width
                if width > max_width:
                    max_width = width
            # Limit maximum width
            self.preview_tree.column(col, width=min(max_width, 200))

    def save_batch_results(self):
        if not hasattr(self, "batch_df"):
            messagebox.showerror("Error", "No results to save")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
        )
        if save_path:
            try:
                self.batch_df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"Results saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {str(e)}")

    def show_analysis(self):
        self.clear_content_area()

        analysis_frame = ttk.Frame(self.content_area, style="Analysis.TFrame")
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(
            analysis_frame, text="Performance Analysis", font=("Segoe UI", 16, "bold")
        ).pack(pady=10, anchor=tk.W)

        # Notebook for different views
        notebook = ttk.Notebook(analysis_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)

        # Tab 1: Prediction History
        history_frame = ttk.Frame(notebook)
        notebook.add(history_frame, text="Prediction History")

        # Treeview for history
        columns = ("date", "student", "score")
        tree = ttk.Treeview(
            history_frame, columns=columns, show="headings", style="Treeview"
        )
        tree.heading("date", text="Date")
        tree.heading("student", text="Student")
        tree.heading("score", text="Predicted Score")

        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Load history
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ph.prediction_date, u.username, ph.predicted_score 
                FROM prediction_history ph
                JOIN users u ON ph.student_username = u.username
                WHERE ph.admin_username = %s
            """,
                (self.current_user,),
            )

            for row in cursor.fetchall():
                tree.insert("", tk.END, values=row)

            cursor.close()
            conn.close()
        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", f"Error fetching history: {err}")

        # Tab 2: Data Distribution
        dist_frame = ttk.Frame(notebook)
        notebook.add(dist_frame, text="Data Distribution")

        # Fetch data for visualization
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            df = pd.read_sql("SELECT * FROM student_data", conn)
            conn.close()
            print(df)

            if not df.empty:
                # Create distribution plots
                fig, axes = plt.subplots(1, 3, figsize=(12, 10))

                # Attendance distribution
                sns.histplot(df["Attendance"], bins=20, kde=True, ax=axes[0])
                axes[0].set_title("Attendance Distribution")

                # Motivation Level impact
                sns.boxplot(
                    x="Motivation_Level", y="Previous_Scores", data=df, ax=axes[1]
                )
                axes[1].set_title("Exam Scores by Motivation Level")

                # Internet Access impact
                sns.boxplot(
                    x="Internet_Access", y="Previous_Scores", data=df, ax=axes[2]
                )
                axes[2].set_title("Exam Scores by Internet Access")

                fig.tight_layout()

                # Embed plot in Tkinter
                canvas = FigureCanvasTkAgg(fig, master=dist_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            else:
                ttk.Label(dist_frame, text="No data available for visualization").pack(
                    pady=20
                )

        except mysql.connector.Error as err:
            messagebox.showerror("Database Error", f"Error fetching data: {err}")

    def logout(self):
        self.current_user = None
        self.current_role = None
        self.show_login_screen()

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def clear_content_area(self):
        for widget in self.content_area.winfo_children():
            widget.destroy()

    def show_batch_graphs(self):
        if not hasattr(self, "batch_df") or self.batch_df.empty:
            messagebox.showerror("Error", "No data available for visualization")
            return

        # Create a new window for graphs
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Batch Data Visualization")
        graph_window.geometry("1200x800")

        # Notebook for different graphs
        notebook = ttk.Notebook(graph_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Predicted Score Distribution
        dist_frame = ttk.Frame(notebook)
        notebook.add(dist_frame, text="Score Distribution")

        # Create figure
        fig1 = plt.Figure(figsize=(10, 5))
        ax1 = fig1.add_subplot(111)

        # Plot histogram
        sns.histplot(self.batch_df["Predicted_Score"], kde=True, ax=ax1, bins=20)
        ax1.set_title("Distribution of Predicted Scores")
        ax1.set_xlabel("Predicted Score")
        ax1.set_ylabel("Frequency")

        # Embed in Tkinter
        canvas1 = FigureCanvasTkAgg(fig1, master=dist_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tab 2: Categorical Comparisons
        cat_frame = ttk.Frame(notebook)
        notebook.add(cat_frame, text="Categorical Comparisons")

        fig2 = plt.Figure(figsize=(10, 8))

        # Create subplots
        ax2 = fig2.add_subplot(211)
        ax3 = fig2.add_subplot(212)

        # Plot 1: Motivation Level vs Predicted Score
        if "Motivation_Level" in self.batch_df.columns:
            sns.boxplot(
                x="Motivation_Level",
                y="Predicted_Score",
                data=self.batch_df,
                order=["Low", "Medium", "High"],
                ax=ax2,
            )
            ax2.set_title("Predicted Scores by Motivation Level")

        # Plot 2: Internet Access vs Predicted Score
        if "Internet_Access" in self.batch_df.columns:
            sns.boxplot(
                x="Internet_Access",
                y="Predicted_Score",
                data=self.batch_df,
                order=["No", "Yes"],
                ax=ax3,
            )
            ax3.set_title("Predicted Scores by Internet Access")

        fig2.tight_layout()

        canvas2 = FigureCanvasTkAgg(fig2, master=cat_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tab 3: Numerical Relationships
        num_frame = ttk.Frame(notebook)
        notebook.add(num_frame, text="Numerical Relationships")

        fig3 = plt.Figure(figsize=(10, 8))

        # Create subplots
        ax4 = fig3.add_subplot(221)
        ax5 = fig3.add_subplot(222)
        ax6 = fig3.add_subplot(223)
        ax7 = fig3.add_subplot(224)

        # Plot 1: Hours Studied vs Predicted Score
        if "Hours_Studied" in self.batch_df.columns:
            sns.scatterplot(
                x="Hours_Studied", y="Predicted_Score", data=self.batch_df, ax=ax4
            )
            ax4.set_title("Hours Studied vs Score")

        # Plot 2: Previous Scores vs Predicted Score
        if "Previous_Scores" in self.batch_df.columns:
            sns.scatterplot(
                x="Previous_Scores", y="Predicted_Score", data=self.batch_df, ax=ax5
            )
            ax5.set_title("Previous Scores vs Predicted Score")

        # Plot 3: Attendance vs Predicted Score
        if "Attendance" in self.batch_df.columns:
            sns.scatterplot(
                x="Attendance", y="Predicted_Score", data=self.batch_df, ax=ax6
            )
            ax6.set_title("Attendance vs Score")

        # Plot 4: Sleep Hours vs Predicted Score
        if "Sleep_Hours" in self.batch_df.columns:
            sns.scatterplot(
                x="Sleep_Hours", y="Predicted_Score", data=self.batch_df, ax=ax7
            )
            ax7.set_title("Sleep Hours vs Score")

        fig3.tight_layout()

        canvas3 = FigureCanvasTkAgg(fig3, master=num_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tab 4: Feature Correlations
        corr_frame = ttk.Frame(notebook)
        notebook.add(corr_frame, text="Feature Correlations")

        fig4 = plt.Figure(figsize=(10, 8))
        ax8 = fig4.add_subplot(111)

        # Calculate correlations for numerical features
        numerical_features = [
            "Hours_Studied",
            "Attendance",
            "Sleep_Hours",
            "Previous_Scores",
            "Tutoring_Sessions",
            "Physical_Activity",
            "Predicted_Score",
        ]

        # Filter only existing numerical columns
        num_cols = [col for col in numerical_features if col in self.batch_df.columns]

        if num_cols:
            corr_df = self.batch_df[num_cols].corr()
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax8)
            ax8.set_title("Correlation Matrix")

        canvas4 = FigureCanvasTkAgg(fig4, master=corr_frame)
        canvas4.draw()
        canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = StudentPerformanceApp(root)
    root.mainloop()