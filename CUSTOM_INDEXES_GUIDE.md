# ➕ Custom Indexes Guide

## ✨ NEW FEATURE: Add/Remove/Customize Indexes

You now have **complete control** over which indexes are calculated! You can:
- ✅ **Remove** default indexes you don't need
- ✅ **Add** custom indexes with your own regional splits
- ✅ **Name** your custom indexes
- ✅ **Configure filters** for each index separately
- ✅ **Preview distribution** for all indexes

## 🎯 Why This Feature?

### **The Problem Before:**
- Fixed set of predefined indexes (e.g., LV FLATS RIGA, LV FLATS PIE-RIGA, etc.)
- Couldn't remove unwanted indexes
- Couldn't create custom regional combinations
- Limited flexibility for your specific analysis needs

### **The Solution Now:**
- **Remove** indexes you don't need (e.g., remove PIE-RIGA if not relevant)
- **Add custom** indexes (e.g., "LV FLATS COASTAL" with Kurzeme + Pierīga)
- **Full control** over what gets calculated

## 📋 How to Use

### Step 1: Enable Individual Index Configuration

1. Check ✅ **"Enable per-category filters"**
2. Expand a category (e.g., "⚙️ Filters for LV FLATS")
3. Select **"Individual indexes"** radio button

### Step 2: Manage Default Indexes

You'll see:

```
🎛️ Index Selection

Default indexes to include:
☑ LV FLATS RIGA
☑ LV FLATS PIE-RIGA
☑ LV FLATS KURZEME + VIDZEME + LATGALE + ZEMGALE
```

**Uncheck any indexes you want to remove**

### Step 3: Add Custom Indexes

```
➕ Add Custom Indexes

Number of custom indexes to add: [0] ← Change to 1, 2, 3, etc.
```

For each custom index:

```
Custom Index 1
┌─────────────────────┬─────────────────────┐
│ Index name:         │ Regions:            │
│ [LV FLATS COASTAL]  │ ☑ Kurzeme          │
│                     │ ☑ Pierīga          │
└─────────────────────┴─────────────────────┘
```

### Step 4: Configure Filters for Each Index

Each index (default or custom) gets its own:
- ☐ Price/m² filter
- ☐ Price filter
- ☐ Area filter
- ☐ Date filter
- ☐ Show distribution plot

### Step 5: Generate Indexes

Click **"🚀 Generate Final Indexes"**

The system will:
1. Skip unchecked default indexes
2. Calculate selected default indexes
3. Calculate your custom indexes
4. Apply filters to each separately

## 💡 Use Cases

### Use Case 1: Remove Unwanted Indexes

**Scenario:** You only need Riga data, not Pieriga

**Solution:**
- Go to "LV FLATS"
- Select "Individual indexes"
- Uncheck ☐ "LV FLATS PIE-RIGA"
- Uncheck ☐ "LV FLATS KURZEME + VIDZEME + LATGALE + ZEMGALE"
- Keep only ☑ "LV FLATS RIGA"

**Result:** Only Riga index is calculated (faster, simpler)

### Use Case 2: Create Regional Groups

**Scenario:** You want "Coastal" vs "Inland" markets

**Solution:**
1. Remove all defaults (uncheck all)
2. Add custom index 1:
   - Name: "LV FLATS COASTAL"
   - Regions: [Kurzeme, Pierīga, Rīga]
3. Add custom index 2:
   - Name: "LV FLATS INLAND"
   - Regions: [Vidzeme, Zemgale, Latgale]

**Result:** Two custom indexes with your regional definitions!

### Use Case 3: Mix Default and Custom

**Scenario:** Keep Riga, but add a custom "Rural" index

**Solution:**
1. Keep ☑ "LV FLATS RIGA" (default)
2. Uncheck others
3. Add custom index:
   - Name: "LV FLATS RURAL"
   - Regions: [Kurzeme, Vidzeme, Latgale, Zemgale]

**Result:** Standard Riga index + your custom rural index

### Use Case 4: Test Different Regional Splits

**Scenario:** Experiment with different combinations

**Solution:**
1. Remove all defaults
2. Try split A:
   - Custom 1: [Rīga]
   - Custom 2: [Pierīga]
   - Custom 3: [All others]
3. Run analysis, export results
4. Try split B:
   - Custom 1: [Rīga + Pierīga]
   - Custom 2: [All others]
5. Compare results!

### Use Case 5: Focus on Specific Municipalities

**Scenario:** You want only major cities

**Solution:**
- Custom index: "LV FLATS MAJOR CITIES"
- Regions: [Rīga] + use municipality filter for specific cities
- Apply strict quality filters
- Get high-quality urban index

## 🎨 UI Elements

### **Default Indexes Selection**
```
Default indexes to include:
☑ LV FLATS RIGA
☐ LV FLATS PIE-RIGA          ← Unchecked = Won't be calculated
☑ LV FLATS KURZEME+...
```

### **Custom Index Creation**
```
Number of custom indexes to add: [2]

Custom Index 1
┌─────────────────────────────┬───────────────────┐
│ Index name:                 │ Regions:          │
│ LV FLATS URBAN              │ ☑ Rīga           │
│                             │ ☑ Pierīga        │
└─────────────────────────────┴───────────────────┘

Custom Index 2
┌─────────────────────────────┬───────────────────┐
│ Index name:                 │ Regions:          │
│ LV FLATS RURAL              │ ☑ Kurzeme        │
│                             │ ☑ Vidzeme        │
│                             │ ☑ Zemgale        │
│                             │ ☑ Latgale        │
└─────────────────────────────┴───────────────────┘
```

### **Index Configuration**
```
🎯 LV FLATS RIGA (Default)
   🌍 Predefined regions: Rīga
   ☐ Price/m² filter
   ☐ Show distribution plot

➕ LV FLATS URBAN (Custom)
   🌍 Custom regions: Rīga + Pierīga
   ☐ Price/m² filter
   ☐ Show distribution plot
```

## ⚙️ Technical Details

### **Index Types**

**Default Indexes:**
- Come from `final_indexes_config`
- Predefined regions (shown as caption, not editable)
- Can be included/excluded via checkboxes
- Marked with 🎯 icon

**Custom Indexes:**
- Created by user
- User-defined names
- User-defined regions (fully editable)
- Marked with ➕ icon

### **Data Flow**

1. **User configures** → Selects defaults + adds customs
2. **Settings stored** → In `per_category_settings`
3. **Generation loop** → Processes only selected/custom indexes
4. **Filters applied** → Individual filters per index
5. **Results displayed** → All indexes in "By Category" tab

### **Storage Structure**

```python
per_category_settings[category] = {
    'filter_level': 'index',
    'selected_defaults': ['LV FLATS RIGA', 'LV FLATS KURZEME+...'],
    'custom_indexes': [
        {'name': 'LV FLATS URBAN', 'regions': ['Rīga', 'Pierīga']},
        {'name': 'LV FLATS RURAL', 'regions': ['Kurzeme', 'Vidzeme', ...]}
    ],
    'index_filters': {
        'LV FLATS RIGA': {...},
        'LV FLATS URBAN': {...},
        ...
    }
}
```

## 📊 Distribution Plots

**Works for both** default and custom indexes!
- Check "Show distribution plot"
- Toggle between Price/m² and Total Price
- See red threshold lines
- Green shaded accepted range
- Statistics: Median, Mean, Min, Max

## ⚠️ Important Notes

### **Minimum One Index**
Each category must have at least one index:
- At least one default selected, OR
- At least one custom index added

### **Unique Names**
Custom index names should be unique. If you create two with the same name, they'll overwrite each other in the results.

### **Regions Required**
Custom indexes MUST have at least one region selected. Empty regions = no data!

### **Property Type Consistency**
Custom indexes use the same property type as their category:
- LV FLATS → Uses Apartments data
- LV HOUSES → Uses Houses data
- etc.

## 🎯 Best Practices

### **1. Start Simple**
Begin with defaults, add customs only when needed

### **2. Meaningful Names**
Use descriptive names:
- ✅ "LV FLATS COASTAL REGIONS"
- ✅ "LV FLATS MAJOR CITIES"
- ❌ "CUSTOM 1"
- ❌ "TEST"

### **3. Logical Groupings**
Group regions that make sense together:
- Economic zones (urban vs. rural)
- Geographic areas (coastal, inland, mountainous)
- Development levels (developed, developing)

### **4. Document Rationale**
If you create custom indexes, document WHY:
- Screenshot the configuration
- Note in your analysis report
- Explain regional grouping logic

### **5. Test Distribution First**
Before finalizing custom indexes:
- Check distribution plot
- Verify sufficient data volume
- Ensure regions have similar characteristics

## 🎉 Benefits

✅ **Complete Flexibility** - Build exactly the indexes you need
✅ **Faster Analysis** - Remove unnecessary indexes
✅ **Custom Insights** - Create market segments that matter to you
✅ **Experimentation** - Try different regional combinations easily
✅ **Professional Output** - Name indexes appropriately for reports

## 📖 Example Workflow

**Goal:** Create Urban vs. Suburban vs. Rural index for apartments

### Configuration:

**Remove all defaults** (uncheck all)

**Add 3 custom indexes:**

1. **LV FLATS URBAN**
   - Regions: [Rīga]
   - Price/m²: 600-5000 EUR

2. **LV FLATS SUBURBAN**
   - Regions: [Pierīga]
   - Price/m²: 500-3000 EUR

3. **LV FLATS RURAL**
   - Regions: [Kurzeme, Vidzeme, Zemgale, Latgale]
   - Price/m²: 300-2000 EUR

### Result:

Three clean, focused indexes that directly compare:
- Urban market dynamics
- Suburban market dynamics
- Rural market dynamics

Perfect for market analysis reports! 📊

---

## 🚀 Quick Start

1. **Refresh browser** (F5)
2. **Enable per-category filters**
3. **Expand LV FLATS** (or any category)
4. **Select "Individual indexes"**
5. **Uncheck unwanted defaults**
6. **Set number of custom indexes** to 1 or 2
7. **Name and configure them**
8. **Click "Generate Final Indexes"**

**Now you can create YOUR perfect set of indexes!** ✨🎯

