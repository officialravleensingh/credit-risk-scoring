import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(figsize=(20, 12))
ax.set_xlim(0, 20)
ax.set_ylim(0, 12)
ax.axis('off')
fig.patch.set_facecolor('white')

# Title
ax.text(10, 11, 'VIDEO TIMELINE - 5 MINUTE PRESENTATION', 
        ha='center', va='center', fontsize=36, weight='bold', color='#1e3a5f')

# Timeline segments
segments = [
    # (start_time, duration, speaker, title, slide, color)
    (0, 30, 'RAVLEEN', 'Opening', '01_title_slide', '#e74c3c'),
    (30, 30, 'RAVLEEN', 'Problem Statement', '02_problem_statement', '#e74c3c'),
    (60, 45, 'ANSH', 'Dataset & EDA', '03_dataset + eda.ipynb', '#3498db'),
    (105, 35, 'ANURAG', 'Preprocessing', '08_preprocessing', '#2ecc71'),
    (140, 55, 'RAVLEEN', 'Model Comparison', '04,13,05,06', '#e74c3c'),
    (195, 25, 'ANSH', 'Feature Importance', '07_feature_importance', '#3498db'),
    (220, 40, 'HIMANSHU', 'Web App Demo', 'app.py LIVE', '#f39c12'),
    (260, 20, 'ANURAG', 'Technical', '09_structure + GitHub', '#2ecc71'),
    (280, 15, 'ANSH', 'Challenges', '10_challenges', '#3498db'),
    (295, 35, 'RAVLEEN', 'Conclusion', '11_future + 12_conclusion', '#e74c3c'),
    (330, 10, 'ALL', 'Closing', '14_end_credits', '#9b59b6')
]

# Draw timeline
y_base = 8
total_duration = 340  # 5:40 in seconds
scale = 18 / total_duration  # Scale to fit width

for start, duration, speaker, title, slide, color in segments:
    x_start = 1 + start * scale
    width = duration * scale
    
    # Draw box
    box = FancyBboxPatch((x_start, y_base - 0.8), width, 1.6,
                         boxstyle="round,pad=0.05", 
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    
    # Add text
    ax.text(x_start + width/2, y_base + 0.4, speaker,
            ha='center', va='center', fontsize=14, weight='bold', color='white')
    ax.text(x_start + width/2, y_base, title,
            ha='center', va='center', fontsize=11, color='white', weight='bold')
    ax.text(x_start + width/2, y_base - 0.4, f'{duration}s',
            ha='center', va='center', fontsize=10, color='white')
    
    # Time markers
    mins = start // 60
    secs = start % 60
    ax.text(x_start, y_base - 1.2, f'{mins}:{secs:02d}',
            ha='center', va='top', fontsize=9, color='#555')

# End time marker
ax.text(19, y_base - 1.2, '5:40',
        ha='center', va='top', fontsize=9, color='#555', weight='bold')

# Legend - Speaker colors
legend_y = 5.5
ax.text(10, legend_y + 1, 'SPEAKER GUIDE', ha='center', va='center',
        fontsize=24, weight='bold', color='#1e3a5f')

speakers_info = [
    ('RAVLEEN SINGH', '#e74c3c', '2 min total', 4),
    ('ANURAG PANDEY', '#2ecc71', '55 sec total', 2),
    ('ANSH TOMAR', '#3498db', '1 min 25 sec', 3),
    ('HIMANSHU CHAUHAN', '#f39c12', '40 sec total', 1)
]

for i, (name, color, duration, segments_count) in enumerate(speakers_info):
    x_pos = 3 + i * 4
    # Color box
    box = FancyBboxPatch((x_pos - 0.8, legend_y - 0.4), 1.6, 0.8,
                         boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(box)
    ax.text(x_pos, legend_y, name, ha='center', va='center',
            fontsize=12, weight='bold', color='white')
    ax.text(x_pos, legend_y - 1, duration, ha='center', va='top',
            fontsize=10, color='#555')
    ax.text(x_pos, legend_y - 1.5, f'{segments_count} segments', ha='center', va='top',
            fontsize=9, color='#777')

# Key slides reference
slides_y = 3
ax.text(10, slides_y + 0.8, 'KEY SLIDES', ha='center', va='center',
        fontsize=20, weight='bold', color='#1e3a5f')

key_slides = [
    '01: Title', '02: Problem', '03: Dataset', '04: Comparison',
    '05: Confusion Matrix', '06: ROC Curves', '07: Features', '08: Pipeline',
    '09: Structure', '10: Challenges', '11: Future', '12: Conclusion',
    '13: Metrics', '14: Credits'
]

for i, slide in enumerate(key_slides):
    row = i // 7
    col = i % 7
    x_pos = 2.5 + col * 2.3
    y_pos = slides_y - row * 0.5
    ax.text(x_pos, y_pos, slide, ha='left', va='center',
            fontsize=9, color='#1e3a5f', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='#ccc'))

# Important notes
notes_y = 0.8
ax.text(10, notes_y, '⚠️  LIVE DEMOS: Jupyter Notebook (Ansh) • Streamlit App (Himanshu) • GitHub (Anurag)',
        ha='center', va='center', fontsize=12, color='#d32f2f', weight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff3cd', edgecolor='#d32f2f', linewidth=2))

# Footer
ax.text(10, 0.2, 'Intelligent Credit Risk Scoring System • Newton School of Technology',
        ha='center', va='center', fontsize=11, color='#777')

plt.tight_layout()
plt.savefig('presentation_assets/TIMELINE_INFOGRAPHIC.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Timeline infographic created: presentation_assets/TIMELINE_INFOGRAPHIC.png")
