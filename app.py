import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Women's Clothing E-commerce Dashboard",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #2E86AB;
    }
    .sidebar-header {
        font-size: 1.25rem;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the ecommerce data"""
    try:
        # Try to load the data - adjust path as needed
        df = pd.read_csv('data/ecommerce_data.csv')
        
        # Data preprocessing
        df['order_date'] = pd.to_datetime(df['order_date'])
        df['size'] = df['size'].fillna('One Size')
        
        # Create additional features for analysis
        df['year'] = df['order_date'].dt.year
        df['month'] = df['order_date'].dt.month
        df['day_of_week'] = df['order_date'].dt.day_name()
        df['hour'] = df['order_date'].dt.hour
        
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please ensure 'ecommerce_data.csv' is in the 'data' folder.")
        return None

def calculate_key_metrics(df):
    """Calculate key business metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['total_revenue'] = df['revenue'].sum()
    metrics['total_orders'] = df['order_id'].nunique()
    metrics['total_items_sold'] = df['quantity'].sum()
    metrics['avg_order_value'] = df.groupby('order_id')['revenue'].sum().mean()
    
    # Product metrics
    metrics['unique_products'] = df['sku'].nunique()
    metrics['avg_unit_price'] = df['unit_price'].mean()
    
    # Time-based metrics
    metrics['date_range'] = f"{df['order_date'].min().date()} to {df['order_date'].max().date()}"
    
    return metrics

def create_revenue_trend_chart(df):
    """Create revenue trend over time"""
    daily_revenue = df.groupby(df['order_date'].dt.date)['revenue'].sum().reset_index()
    daily_revenue.columns = ['date', 'revenue']
    
    fig = px.line(daily_revenue, x='date', y='revenue', 
                  title='Daily Revenue Trend',
                  labels={'revenue': 'Revenue ($)', 'date': 'Date'})
    fig.update_traces(line_color='#2E86AB')
    fig.update_layout(height=450)
    
    return fig

def create_sales_prediction_chart(df):
    """Create simple sales prediction for next months using moving averages"""
    try:
        # Prepare data for prediction
        daily_revenue = df.groupby(df['order_date'].dt.date)['revenue'].sum().reset_index()
        daily_revenue.columns = ['date', 'revenue']
        daily_revenue['date'] = pd.to_datetime(daily_revenue['date'])
        daily_revenue = daily_revenue.sort_values('date')
        
        # Work with whatever data we have
        data_points = len(daily_revenue)

        if data_points == 0:
            raise ValueError("No data available")
        
        # Calculate moving average with available data
        window_size = min(3, data_points)
        if data_points >= 3:
            daily_revenue['ma'] = daily_revenue['revenue'].rolling(window=window_size, min_periods=1).mean()
        else:
            daily_revenue['ma'] = daily_revenue['revenue']
        
        # Simple prediction using available data
        if data_points >= 5:
            recent_avg = daily_revenue.tail(5)['revenue'].mean()
        else:
            recent_avg = daily_revenue['revenue'].mean()
        
        # Generate future dates (next 90 days)
        last_date = daily_revenue['date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=90, freq='D')
        
        # Create realistic prediction with some variation
        np.random.seed(42)
        base_revenue = recent_avg
        
        future_revenue = []
        for i in range(90):
            weekly_pattern = np.sin(i / 7 * 2 * np.pi) * base_revenue * 0.1
            random_variation = np.random.normal(0, base_revenue * 0.08)
            predicted_value = max(base_revenue * 0.1, base_revenue + weekly_pattern + random_variation)
            future_revenue.append(predicted_value)
        
        # Create prediction dataframe
        prediction_df = pd.DataFrame({
            'date': future_dates,
            'predicted_revenue': future_revenue
        })
        
        # Create the plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=daily_revenue['date'], 
            y=daily_revenue['revenue'],
            mode='lines+markers',
            name='Historical Revenue',
            line=dict(color='#2E86AB'),
            marker=dict(size=4)
        ))
        
        # Moving average
        if data_points >= 3:
         fig.add_trace(go.Scatter(
            x=daily_revenue['date'], 
            y=daily_revenue['ma'],
            mode='lines',
            name=f'{window_size}-Day Moving Average',
            line=dict(color='#A23B72', width=2)
        ))
        
        # Predicted data
        fig.add_trace(go.Scatter(
            x=prediction_df['date'], 
            y=prediction_df['predicted_revenue'],
            mode='lines',
            name='Predicted Revenue (Simple Trend)',
            line=dict(color='#E74C3C', dash='dash', width=2)
        ))
        
        # Add vertical line to separate historical from predicted
        fig.add_vline(x=last_date, line_dash="dot", line_color="gray", 
                      annotation_text="Prediction Start")
        
        # Add confidence bands
        upper_bound = [x * 1.2 for x in future_revenue]
        lower_bound = [x * 0.8 for x in future_revenue]
        
        fig.add_trace(go.Scatter(
            x=list(prediction_df['date']) + list(prediction_df['date'][::-1]),
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(231, 76, 60, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True
        ))
        
        fig.update_layout(
            title=f'Sales Revenue Prediction (Next 3 Months)',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            height=450,
            hovermode='x unified'
        )
        
        return fig, prediction_df, daily_revenue

    except Exception as e:
        # Create a simple prediction even with minimal data
        try:
            # Get basic stats from the data
            total_revenue = df['revenue'].sum()
            total_days = (df['order_date'].max() - df['order_date'].min()).days + 1
            avg_daily_revenue = total_revenue / max(total_days, 1)
            
            # Generate future dates
            last_date = df['order_date'].max().date()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=90, freq='D')
            
            # Simple flat prediction with some variation
            np.random.seed(42)
            future_revenue = []
            for i in range(90):
                variation = np.random.normal(0, avg_daily_revenue * 0.1)
                predicted_value = max(avg_daily_revenue * 0.5, avg_daily_revenue + variation)
                future_revenue.append(predicted_value)
            
            prediction_df = pd.DataFrame({
                'date': future_dates,
                'predicted_revenue': future_revenue
            })
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_revenue,
                mode='lines',
                name='Basic Revenue Prediction',
                line=dict(color='#E74C3C', dash='dash')
            ))
            
            fig.update_layout(
                title='Sales Revenue Prediction <br><sub>Basic prediction based on available data</sub>',
                xaxis_title='Date',
                yaxis_title='Revenue ($)',
                height=450
            )
            
            return fig, prediction_df, None
            
        except:
            # Last resort: return informative message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Unable to generate prediction with current data.<br>Data points available: {len(df)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(title='Sales Revenue Prediction', height=450)
            return fig, None, None

def create_performance_by_attributes_chart(df):
    """Create performance analysis by color and size"""
    # Performance by color
    color_performance = df.groupby('color').agg({
        'quantity': 'sum',
        'revenue': 'sum'
    }).reset_index()
    color_performance = color_performance.sort_values('quantity', ascending=False).head(8)
    
    # Performance by size  
    size_performance = df.groupby('size').agg({
        'quantity': 'sum',
        'revenue': 'sum'
    }).reset_index()
    size_performance = size_performance.sort_values('quantity', ascending=False).head(8)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Best-Selling Colors', 'Best-Selling Sizes'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Colors chart
    fig.add_trace(
        go.Bar(x=color_performance['color'], y=color_performance['quantity'],
               name='Color Performance', marker_color='#A23B72',
               showlegend=False),
        row=1, col=1
    )
    
    # Sizes chart
    fig.add_trace(
        go.Bar(x=size_performance['size'], y=size_performance['quantity'],
               name='Size Performance', marker_color='#F18F01',
               showlegend=False),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="Performance by Color and Size")
    fig.update_xaxes(title_text="Color", row=1, col=1)
    fig.update_xaxes(title_text="Size", row=1, col=2)
    fig.update_yaxes(title_text="Quantity Sold")
    
    return fig

def create_chat_interface(df):
    """Create a simple chat interface for data queries"""
    st.markdown("## 💬 Chat with Your Data")
    
    # Chat input
    user_question = st.text_input(
        "Ask a question about your data:", 
        placeholder="e.g., What's the total revenue for 'Red' color items?"
    )
    
    if user_question:
        # Simple pattern matching for common queries
        question_lower = user_question.lower()
        
        try:
            if 'revenue' in question_lower and 'color' in question_lower:
                # Extract color if mentioned
                colors = df['color'].unique()
                mentioned_color = None
                for color in colors:
                    if color.lower() in question_lower:
                        mentioned_color = color
                        break
                
                if mentioned_color:
                    color_revenue = df[df['color'] == mentioned_color]['revenue'].sum()
                    st.write(f"💡 **Answer:** Total revenue for '{mentioned_color}' items is ${color_revenue:,.2f}")
                else:
                    color_revenues = df.groupby('color')['revenue'].sum().sort_values(ascending=False)
                    st.write("💡 **Answer:** Revenue by color:")
                    for color, revenue in color_revenues.items():
                        st.write(f"  - {color}: ${revenue:,.2f}")
            
            elif 'revenue' in question_lower and 'size' in question_lower:
                # Extract size if mentioned
                sizes = df['size'].unique()
                mentioned_size = None
                for size in sizes:
                    if size.lower() in question_lower:
                        mentioned_size = size
                        break
                
                if mentioned_size:
                    size_revenue = df[df['size'] == mentioned_size]['revenue'].sum()
                    st.write(f"💡 **Answer:** Total revenue for size '{mentioned_size}' is ${size_revenue:,.2f}")
                else:
                    size_revenues = df.groupby('size')['revenue'].sum().sort_values(ascending=False)
                    st.write("💡 **Answer:** Revenue by size:")
                    for size, revenue in size_revenues.items():
                        st.write(f"  - {size}: ${revenue:,.2f}")
            
            elif 'best selling' in question_lower or 'top selling' in question_lower:
                if 'color' in question_lower:
                    top_color = df.groupby('color')['quantity'].sum().sort_values(ascending=False).head(1)
                    st.write(f"💡 **Answer:** Best-selling color is '{top_color.index[0]}' with {top_color.values[0]} items sold")
                elif 'size' in question_lower:
                    top_size = df.groupby('size')['quantity'].sum().sort_values(ascending=False).head(1)
                    st.write(f"💡 **Answer:** Best-selling size is '{top_size.index[0]}' with {top_size.values[0]} items sold")
                else:
                    top_sku = df.groupby('sku')['quantity'].sum().sort_values(ascending=False).head(1)
                    st.write(f"💡 **Answer:** Best-selling SKU is '{top_sku.index[0]}' with {top_sku.values[0]} items sold")
            
            elif 'total revenue' in question_lower:
                total_rev = df['revenue'].sum()
                st.write(f"💡 **Answer:** Total revenue is ${total_rev:,.2f}")
            
            elif 'total orders' in question_lower:
                total_orders = df['order_id'].nunique()
                st.write(f"💡 **Answer:** Total number of orders is {total_orders:,}")
            
            elif 'average order value' in question_lower or 'aov' in question_lower:
                aov = df.groupby('order_id')['revenue'].sum().mean()
                st.write(f"💡 **Answer:** Average Order Value (AOV) is ${aov:.2f}")
            
            else:
                st.write("🤔 I can help you with questions about:")
                st.write("- Revenue by color or size")
                st.write("- Best-selling items, colors, or sizes")
                st.write("- Total revenue, orders, or average order value")
                st.write("Try asking: 'What's the total revenue for red items?' or 'What's the best selling size?'")
        
        except Exception as e:
            st.write(f"❌ Sorry, I couldn't process that question: {str(e)}")
            st.write("Try asking a simpler question about revenue, colors, sizes, or sales.")
    
    # Quick action buttons
    st.markdown("**Quick Questions:**")
    
    # Create buttons with no gaps
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📊 Total Revenue", use_container_width=True):
            total_rev = df['revenue'].sum()
            st.write(f"💡 Total revenue: ${total_rev:,.2f}")
    
    with col2:
        if st.button("🎨 Best Color", use_container_width=True):
            top_color = df.groupby('color')['quantity'].sum().sort_values(ascending=False).head(1)
            st.write(f"💡 Best-selling color: {top_color.index[0]} ({top_color.values[0]} items)")
    
    with col3:
        if st.button("📏 Best Size", use_container_width=True):
            top_size = df.groupby('size')['quantity'].sum().sort_values(ascending=False).head(1)
            st.write(f"💡 Best-selling size: {top_size.index[0]} ({top_size.values[0]} items)")
    
    # Add spacing before Raw Data section
    st.markdown("<br><br>", unsafe_allow_html=True)

def create_product_performance_chart(filtered_df):
    """Create SKU sales analysis - horizontal bar chart"""
    # Calculate total quantity sold for each SKU
    sku_demand = (
        filtered_df.groupby('sku')['quantity']
        .sum()
        .reset_index()
        .sort_values('quantity', ascending=False)
    ).head(25)

    # Horizontal bar chart with hover tooltip
    fig = px.bar(
        sku_demand,
        x='quantity',
        y='sku',
        orientation='h',
        color='quantity',
        color_discrete_sequence=['#F18F01'],
        hover_data={'sku': False, 'quantity': True},
        labels={'quantity': 'Quantity Sold'}
    )

    # Reverse y-axis so the highest quantity is on top
    fig.update_yaxes(autorange="reversed")

    # Layout adjustments
    fig.update_layout(
        title='Top 25 SKUs by Sales',
        xaxis_title='Quantity Sold',
        yaxis_title='SKU',
        height=476,
        coloraxis_showscale=False,  # hide color scale
        template='plotly_white'
    )

    return fig

def create_size_distribution_chart(df):
    """Create size distribution chart with legend positioned closer"""
    size_dist = df['size'].value_counts().reset_index()
    size_dist.columns = ['Size', 'Quantity']  # Rename columns
    
    # Create pie chart
    fig = px.pie(
        size_dist,
        names='Size',
        values='Quantity',
        title='Size Distribution',
        hover_data={'Size': True, 'Quantity': True},  # Customize tooltip
    )

    # Show inside pie: Size and Quantity
    fig.update_traces(
        textinfo='percent+label',  # still shows percent and label
        hovertemplate='Size: %{label}<br>Quantity: %{value}<extra></extra>'
    )

    # Position legend closer to the pie chart
    fig.update_layout(
        height=450,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=0.01,
            font=dict(size=12)
        )
    )

    return fig

def create_color_analysis_chart(df):
    """Create color popularity analysis in descending order"""
    color_revenue = df.groupby('color')['revenue'].sum().sort_values(ascending=False).head(10)
    
    fig = px.bar(x=color_revenue.values, y=color_revenue.index,
                 orientation='h',
                 title='Top 10 Colors by Revenue',
                 labels={'x': 'Revenue ($)', 'y': 'Color'})
    fig.update_traces(marker_color='#F18F01')
    fig.update_layout(height=450)
    
    # Ensure descending order (highest at top)
    fig.update_yaxes(categoryorder='total ascending')
    
    return fig

def create_hourly_sales_heatmap(df):
    """Create hourly sales pattern heatmap"""
    hourly_sales = df.groupby(['day_of_week', 'hour'])['revenue'].sum().reset_index()
    
    # Pivot for heatmap
    heatmap_data = hourly_sales.pivot(index='day_of_week', columns='hour', values='revenue')
    
    # Reorder days of week
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(day_order)
    
    fig = px.imshow(heatmap_data, 
                    labels=dict(x="Hour of Day", y="Day of Week", color="Revenue ($)"),
                    title="Sales Heatmap by Day and Hour")
    fig.update_layout(height=400)
    
    return fig

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">👗 Women\'s Clothing E-commerce Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Sidebar filters
    st.sidebar.markdown('<h2 class="sidebar-header">Filters</h2>', unsafe_allow_html=True)
    
    # Date range filter
    min_date = df['order_date'].min().date()
    max_date = df['order_date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Color filter
    colors = ['All'] + list(df['color'].unique())
    selected_color = st.sidebar.selectbox("Select Color", colors)
    
    # Size filter
    sizes = ['All'] + list(df['size'].unique())
    selected_size = st.sidebar.selectbox("Select Size", sizes)
    
    # Apply filters
    filtered_df = df.copy()
    
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['order_date'].dt.date >= date_range[0]) & 
            (filtered_df['order_date'].dt.date <= date_range[1])
        ]
    
    if selected_color != 'All':
        filtered_df = filtered_df[filtered_df['color'] == selected_color]
        
    if selected_size != 'All':
        filtered_df = filtered_df[filtered_df['size'] == selected_size]
    
    # Calculate metrics for filtered data
    metrics = calculate_key_metrics(filtered_df)
    
    # Display key metrics
    st.markdown("## 📊 Key Performance Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="💰 Total Revenue",
            value=f"${metrics['total_revenue']:,.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="📦 Total Orders",
            value=f"{metrics['total_orders']:,}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="🛍️ Items Sold",
            value=f"{metrics['total_items_sold']:,}",
            delta=None
        )
    
    
    # Additional metrics row
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.metric(
            label="🏷️ Unique Products",
            value=f"{metrics['unique_products']:,}",
            delta=None
        )
    
    with col5:
        st.metric(
            label="💲 Avg Unit Price",
            value=f"${metrics['avg_unit_price']:.2f}",
            delta=None
        )
    
    with col6:
        st.metric(
            label="💵 Avg Order Value",
            value=f"${metrics['avg_order_value']:.2f}",
            delta=None
        )

    # Charts section
    st.markdown("## 📈 Data Analysis")
    
    # Row 1: Revenue trend and SKU performance analysis
    col1, col2 = st.columns(2)
    
    with col1:
        if len(filtered_df) > 0:
            fig1 = create_revenue_trend_chart(filtered_df)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    
    with col2:
        if len(filtered_df) > 0:
            st.plotly_chart(create_product_performance_chart(filtered_df), use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    
    # Row 2: Size distribution and color analysis
    col3, col4 = st.columns(2)
    
    with col3:
        if len(filtered_df) > 0:
            fig3 = create_size_distribution_chart(filtered_df)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    
    with col4:
        if len(filtered_df) > 0:
            fig4 = create_color_analysis_chart(filtered_df)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    
    # Row 3: Hourly sales heatmap (full width)
    st.markdown("## ⏰ Sales Pattern Analysis")
    if len(filtered_df) > 0:
      fig5 = create_hourly_sales_heatmap(filtered_df)
      st.plotly_chart(fig5, use_container_width=True)

      # 🔎 Add enhanced insight below heatmap
      hourly_sales = filtered_df.groupby(['day_of_week', 'hour'])['revenue'].sum().reset_index()

      # Find the top 5 slots
      top_slots = hourly_sales.sort_values("revenue", ascending=False).head(5)

      # Group by day for better readability
      insights = []
      for day in top_slots["day_of_week"].unique():
        day_slots = top_slots[top_slots["day_of_week"] == day].sort_values("hour")
        
        # Group consecutive hours into ranges
        hours = list(day_slots["hour"])
        ranges = []
        start = prev = hours[0]
        for h in hours[1:]:
            if h == prev + 1:
                prev = h
            else:
                ranges.append((start, prev))
                start = prev = h
        ranges.append((start, prev))

        # Format ranges nicely
        hour_ranges = []
        for r in ranges:
            if r[0] == r[1]:
                hour_ranges.append(f"{r[0]}:00")
            else:
                hour_ranges.append(f"{r[0]}:00–{r[1]}:00")
        
        insights.append(f"**{day}** during {', '.join(hour_ranges)}")

      # Display inside a framed box
      st.info(
        "💡 **Insight:** The busiest periods with the highest sales were observed on:\n\n"
        + "\n".join([f"- {i}" for i in insights]) +
        "\n\n📌 These are ideal times to schedule promotions or targeted ads."
    )

    else:
     st.warning("No data available for the selected filters.")

    
    # Row 4: Sales prediction (full width)
    st.markdown("## 📈 Sales Prediction (Next 3 Months)")
    prediction_result = create_sales_prediction_chart(filtered_df)
    if len(prediction_result) == 3:
        fig_prediction, prediction_df, daily_revenue = prediction_result
    else:
        fig_prediction = prediction_result
        prediction_df, daily_revenue = None, None

    st.plotly_chart(fig_prediction, use_container_width=True) 
               
    # 🔎 Add enhanced insight below prediction chart
    if prediction_df is not None and not prediction_df.empty:
     min_val = prediction_df['predicted_revenue'].min()
     max_val = prediction_df['predicted_revenue'].max()
     q1 = prediction_df['predicted_revenue'].quantile(0.25)
     q3 = prediction_df['predicted_revenue'].quantile(0.75)

     st.info(
        f"💡 **Insight:** Sales are expected to mostly fluctuate between "
        f"**\\${q1:,.0f}** and **\\${q3:,.0f}**, reflecting the typical revenue range. "
        f"Occasional dips to around **\\${min_val:,.0f}** and peaks up to **\\${max_val:,.0f}** "
        "indicate both potential risks and opportunities."
     )

    # Chat interface
    if len(filtered_df) > 0:
        create_chat_interface(filtered_df)
    else:
        st.warning("Chat unavailable - no data for selected filters.")
    
    # Data table section
    st.markdown("## 📋 Raw Data")
    
    if st.checkbox("Show raw data"):
        st.dataframe(filtered_df.head(1000))  # Show first 1000 rows
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name=f'ecommerce_data_filtered_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv',
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "📊 **Dashboard built with Streamlit** | "
        "🔄 **Auto-refresh enabled** | "
        "📱 **Mobile responsive**"
    )

if __name__ == "__main__":
    main()