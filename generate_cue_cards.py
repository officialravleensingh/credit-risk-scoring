import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import os

os.makedirs('presentation_assets/cue_cards', exist_ok=True)

# Speaker data
speakers_data = {
    'RAVLEEN': {
        'color': '#e74c3c',
        'segments': [
            ('Opening', '0:00-0:30', '01_title_slide', 'Introduce team and project'),
            ('Problem', '0:30-1:00', '02_problem_statement', 'Emphasize 90.15% accuracy'),
            ('Models', '2:20-3:15', '04,13,05,06', 'RF wins! 99% recall'),
            ('Conclusion', '4:55-5:30', '11,12', 'Future work + demo URL')
        ],
        'total': '~2 minutes',
        'key_points': ['90.15% accuracy', '99% recall', 'Best ROC-AUC', 'Demo URL']
    },
    'ANURAG': {
        'color': '#2ecc71',
        'segments': [
            ('Preprocessing', '1:45-2:20', '08_preprocessing', '3 steps: Encode→Scale→Split'),
            ('Technical', '4:20-4:40', '09_structure + GitHub', 'Modular structure + version control')
        ],
        'total': '~55 seconds',
        'key_points': ['Label encoding', 'Standard scaling', '80-20 split', 'GitHub commits']
    },
    'ANSH': {
        'color': '#3498db',
        'segments': [
            ('Dataset', '1:00-1:45', '03 + eda.ipynb', '20K samples, credit score = top'),
            ('Features', '3:15-3:40', '07_feature_importance', 'Credit score (0.28) highest'),
            ('Challenges', '4:40-4:55', '10_challenges', '4 challenges → 4 solutions')
        ],
        'total': '~1 min 25 sec',
        'key_points': ['20,000 samples', '21 features', 'Credit score top', 'No missing values']
    },
    'HIMANSHU': {
        'color': '#f39c12',
        'segments': [
            ('Web App', '3:40-4:20', 'app.py LIVE', 'Demo: 35yo, $60K, 720 credit → Low Risk 92%')
        ],
        'total': '~40 seconds',
        'key_points': ['4 input sections', 'Sidebar metrics', 'Real-time prediction', 'Streamlit Cloud']
    }
}

# Generate cue card for each speaker
for speaker, data in speakers_data.items():
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    
    # Header
    header_box = Rectangle((0, 12.5), 10, 1.5, facecolor=data['color'], edgecolor='black', linewidth=3)
    ax.add_patch(header_box)
    ax.text(5, 13.25, f'{speaker} SINGH' if speaker == 'RAVLEEN' else f'{speaker} {"PANDEY" if speaker == "ANURAG" else "TOMAR" if speaker == "ANSH" else "CHAUHAN"}',
            ha='center', va='center', fontsize=32, weight='bold', color='white')
    
    # Total time
    ax.text(5, 11.8, f'Total Speaking Time: {data["total"]}',
            ha='center', va='center', fontsize=18, weight='bold', color=data['color'])
    
    # Segments
    y_pos = 11
    ax.text(5, y_pos, 'YOUR SEGMENTS', ha='center', va='center',
            fontsize=20, weight='bold', color='#1e3a5f')
    y_pos -= 0.8
    
    for i, (title, time, slide, note) in enumerate(data['segments'], 1):
        # Segment box
        box = FancyBboxPatch((0.5, y_pos - 1.8), 9, 1.6,
                            boxstyle="round,pad=0.1",
                            facecolor='#f5f5f5', edgecolor=data['color'], linewidth=3)
        ax.add_patch(box)
        
        # Segment number
        circle = plt.Circle((1.2, y_pos - 1), 0.4, color=data['color'], ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(1.2, y_pos - 1, str(i), ha='center', va='center',
                fontsize=20, weight='bold', color='white')
        
        # Segment details
        ax.text(2, y_pos - 0.6, f'{title} ({time})', ha='left', va='center',
                fontsize=16, weight='bold', color='#1e3a5f')
        ax.text(2, y_pos - 1.1, f'Slide: {slide}', ha='left', va='center',
                fontsize=13, color='#555', family='monospace')
        ax.text(2, y_pos - 1.5, f'Note: {note}', ha='left', va='center',
                fontsize=12, color='#777', style='italic')
        
        y_pos -= 2.2
    
    # Key points
    y_pos -= 0.5
    ax.text(5, y_pos, 'KEY POINTS TO MENTION', ha='center', va='center',
            fontsize=18, weight='bold', color='#1e3a5f')
    y_pos -= 0.6
    
    for point in data['key_points']:
        ax.text(1, y_pos, f'✓ {point}', ha='left', va='center',
                fontsize=14, color='#2e7d32', weight='bold')
        y_pos -= 0.5
    
    # Recording tips
    y_pos -= 0.5
    tips_box = FancyBboxPatch((0.5, y_pos - 2.5), 9, 2.3,
                             boxstyle="round,pad=0.1",
                             facecolor='#fff3cd', edgecolor='#d32f2f', linewidth=2)
    ax.add_patch(tips_box)
    
    ax.text(5, y_pos - 0.3, '⚠️ RECORDING TIPS', ha='center', va='center',
            fontsize=16, weight='bold', color='#d32f2f')
    
    tips = [
        '1. Practice 3-4 times before recording',
        '2. Speak clearly, not too fast',
        '3. Keep slides visible 5+ seconds',
        '4. Pause 1 second between slides',
        '5. Stay hydrated, have fun!'
    ]
    
    y_pos -= 0.7
    for tip in tips:
        ax.text(1, y_pos, tip, ha='left', va='center',
                fontsize=11, color='#555')
        y_pos -= 0.35
    
    # Footer
    ax.text(5, 0.5, 'Credit Risk Scoring System • Newton School of Technology',
            ha='center', va='center', fontsize=10, color='#999')
    
    plt.tight_layout()
    plt.savefig(f'presentation_assets/cue_cards/{speaker}_CUE_CARD.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Cue card created for {speaker}")

print("\n✅ All cue cards generated successfully!")
print("📁 Location: presentation_assets/cue_cards/")
