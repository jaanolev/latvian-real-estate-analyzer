# 📊 Distribution Preview Guide

## ✨ NEW FEATURE: Live Filter Effect Visualization

When configuring **individual index filters**, you can now see a **live distribution plot** showing exactly how your filter settings affect the data!

## 🎯 What This Shows You

### **Interactive Distribution Histogram**
- **50-bin histogram** of all transactions for that index
- **Toggle** between "Price per m²" and "Total Price" distributions
- **Vertical red lines** marking your filter min/max thresholds
- **Green shaded area** showing the accepted range
- **Real-time updates** as you adjust sliders

### **Filter Effect Statistics**
- **Records kept/removed**: "1,500 / 2,000 records (75% kept, 25% removed)"
- **Distribution stats**: Median, Mean, Min, Max values
- **Visual feedback**: See immediately if filters are too strict or too loose

## 📋 How to Use

### Step 1: Enable Individual Index Filtering

1. Enable **"per-category filters"** checkbox
2. Expand a category (e.g., "⚙️ Filters for LV FLATS")
3. Select **"Individual indexes"** radio button

### Step 2: Configure an Index

For any index (e.g., "🎯 LV FLATS RIGA"):
1. Set your filters (Price/m², Price, Area, Date, etc.)
2. Check ☑️ **"Show distribution plot"**

### Step 3: View the Distribution

You'll see:

```
📊 Filter Effect Preview
☑ Show distribution plot

Radio: (•) Price per m²  ( ) Total Price

[Histogram with vertical lines and shaded area]

📊 Filter Effect: 1,850 / 2,000 records (92.5% kept, 7.5% removed)

[Median]  [Mean]   [Min]    [Max]
  750      820      50      5000
```

### Step 4: Adjust and Observe

- **Move the sliders** → Lines and shading update
- **Change price/m² min/max** → See how many records are affected
- **Toggle to "Total Price"** → View price distribution instead
- **Statistics update** → See median, mean, min, max change

## 🎨 Visual Elements

### **Red Dashed Lines**
- Show your **filter thresholds** (min and max)
- Labeled with values: "Min: 500" and "Max: 5000"
- Help you see where your cutoffs are

### **Green Shaded Area**
- Shows the **accepted range** (data that passes your filter)
- Everything outside this area will be removed
- Makes it easy to see what you're keeping vs. removing

### **Histogram Bars**
- Show how data is **distributed**
- Tall bars = many transactions at that price
- Short bars = few transactions
- Helps identify outliers and typical values

## 💡 Use Cases

### Use Case 1: Avoid Over-Filtering

**Scenario**: You set filters but remove too much data

**Solution**: 
- Check distribution plot
- If 50%+ removed, you might be too strict
- Adjust min/max to keep more representative data

### Use Case 2: Identify Outliers

**Scenario**: You see extreme values in your index

**Solution**:
- View distribution plot
- See if there are outliers far from the main distribution
- Set thresholds to exclude unrealistic prices

### Use Case 3: Market-Appropriate Thresholds

**Scenario**: Different regions have different typical prices

**Solution**:
- LV FLATS RIGA: View distribution → Set 800-5000 EUR/m²
- LV FLATS KURZEME: View distribution → Set 300-2000 EUR/m²
- Each index gets appropriate thresholds for its market

### Use Case 4: Data Quality Check

**Scenario**: You want to verify data quality before analysis

**Solution**:
- View distribution for each index
- Check for suspicious patterns (e.g., many values at exactly 1000)
- Identify data entry errors or anomalies

## 📊 Understanding the Plot

### **Normal Distribution**
```
    Frequency
      ▲
      |     ┌─┐
      |   ┌─┘ └─┐
      |  ┌┘     └┐
      | ┌┘       └┐
      └─┴─────────┴─► Price/m²
     Low  Mid   High
```
Most data in the middle, few extremes = Good quality!

### **Skewed Distribution**
```
    Frequency
      ▲
      | ┌─┐
      | │ └─┐
      | │   └─┐
      | │     └──┐
      └─┴─────────┴─► Price/m²
     Low  Mid   High
```
Most data clustered low, few high values = Watch for outliers!

### **Bimodal Distribution**
```
    Frequency
      ▲
      | ┌─┐   ┌─┐
      | │ │   │ │
      | │ │   │ │
      | │ │   │ │
      └─┴─┴───┴─┴─► Price/m²
     Low  Mid High
```
Two peaks = Might have two different markets mixed!

## ⚙️ Technical Details

### **Data Source**
- Loads actual transaction data for the property type
- Filters to the index's predefined regions
- Applies per-index region filter (if set)
- Shows ALL historical data (not just selected date range)

### **Calculation**
- **Price per m²**: Uses Total_EUR_m2 or Land_EUR_m2 column (or calculates)
- **Total Price**: Uses Price_EUR column
- **Clean data only**: Removes NaN, zero, and negative values

### **Performance**
- Data is loaded on-demand (only when "Show distribution" is checked)
- Efficient for large datasets
- Cached by Streamlit for fast re-rendering

## 📈 Statistics Explained

### **Median**
- **Middle value** when data is sorted
- Less affected by outliers than mean
- Good indicator of "typical" price

### **Mean**
- **Average** of all values
- Affected by extreme values
- Higher than median = skewed by high prices

### **Min/Max**
- **Extreme values** in the dataset
- Help identify data quality issues
- Very low min (<10) might indicate errors

## 🎯 Best Practices

### 1. **Start with No Filters**
- View raw distribution first
- Identify natural data range
- Spot obvious outliers

### 2. **Set Realistic Thresholds**
- Don't cut off >30% of data unless justified
- Keep thresholds within main distribution peak
- Document reasons for aggressive filtering

### 3. **Compare Before/After**
- Note the "X% kept, Y% removed" statistic
- Verify you're not removing valid data
- Check if removed data is truly outliers

### 4. **Use Both Views**
- **Price per m²**: Best for comparing across property sizes
- **Total Price**: Shows absolute value distribution
- Different insights from each view

### 5. **Regional Context**
- Riga: Higher prices normal, wider range acceptable
- Rural: Lower prices normal, tighter range may be appropriate
- Consider local market characteristics

## ⚠️ Warnings

### **Large Removal Percentages**
If you see "> 30% removed", ask yourself:
- Are these truly outliers?
- Or am I filtering out valid transactions?
- Should I relax the thresholds?

### **No Data Between Lines**
If the green shaded area has no histogram bars:
- Your thresholds are in a data gap
- No transactions will pass this filter!
- Adjust min/max to include actual data

### **Extreme Skew**
If 95% of data is bunched at one end:
- Consider using percentile-based thresholds
- Or use log scale for viewing
- Might indicate data quality issues

## 🎉 Benefits

✅ **Immediate Visual Feedback** - See filter effects instantly
✅ **Data-Driven Decisions** - Set thresholds based on actual distribution
✅ **Avoid Over-Filtering** - Prevent removing too much data
✅ **Quality Control** - Spot data anomalies visually
✅ **Documentation** - Screenshot plots for reports
✅ **Transparency** - Show stakeholders why filters were chosen

---

**Now you can see EXACTLY what your filters do before applying them!** 📊✨

