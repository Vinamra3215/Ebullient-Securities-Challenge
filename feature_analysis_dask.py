import dask_cudf
import cudf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
import pandas as pd
import re
from collections import defaultdict
from filter import get_pairs_for_day

def analyze_feature(data_df, price_pairs=None, selected_features=None):
    warnings.filterwarnings('ignore')

    # Handle different dataframe types
    if isinstance(data_df, dask_cudf.DataFrame):
        data_df = data_df.compute().to_pandas()
    elif isinstance(data_df, cudf.DataFrame):
        data_df = data_df.to_pandas()
    elif isinstance(data_df, pd.DataFrame):
        pass  # Already pandas, no conversion needed
    else:
        raise TypeError(f"Unsupported dataframe type: {type(data_df)}")

    data_df["Time"] = pd.to_timedelta(data_df["Time"])
    data_df["Day"] = (data_df["Time"].diff() < pd.Timedelta(0)).cumsum()
    data_df = data_df.dropna(subset=["Price"]).reset_index(drop=True)
    data_df["PlotIndex"] = range(len(data_df))

    min_price = data_df["Price"].min()
    max_price = data_df["Price"].max()
    price_range = max_price - min_price
    
    feature_columns = [c for c in data_df.columns if c not in ['Time', 'Price', 'Day', 'PlotIndex']]
    # Filter to selected features if specified
    if selected_features is not None:
        feature_columns = [c for c in feature_columns if c in selected_features]
        print(f"  Selected {len(feature_columns)} features out of {len([c for c in data_df.columns if c not in ['Time', 'Price', 'Day', 'PlotIndex']])} available")
        if len(feature_columns) == 0:
            print(f"  Warning: No matching features found!")
    else:
        print(f"  Using all {len(feature_columns)} features")

    normalized_feature_columns = {} # Dictionary to store names {original: normalized}

    for col in feature_columns:
        print(f"  Normalizing feature '{col}'...")
        col_min = data_df[col].min()
        col_max = data_df[col].max()
        col_range = col_max - col_min
        norm_col_name = f"{col}_norm"
        normalized_feature_columns[col] = norm_col_name # Store mapping
        data_df[norm_col_name] = data_df[col]
        # if col_range > 1e-9: # Avoid division by zero
        #     data_df[norm_col_name] = (data_df[col] - col_min) / col_range
        #     print(f"    Created normalized column: {norm_col_name} (Min={col_min:.4f}, Max={col_max:.4f})")
        # else: # Constant value column
        #     data_df[norm_col_name] = 0.5 # Assign midpoint
        #     print(f"    Created normalized column: {norm_col_name} (Constant value, set to 0.5)")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot Price (Primary Y-axis) - Day by Day (disconnected)
    for day, df_day in data_df.groupby("Day"):
        time_strings = df_day["Time"].apply(lambda x: str(x).split()[-1]).tolist()
        fig.add_trace(go.Scatter(
            x=df_day["PlotIndex"],
            y=df_day["Price"],
            mode='lines',
            name=f"Price Day {day}", # Name for hover, not legend
            line=dict(width=2, color='rgba(0,0,0,0.7)'), # Consistent dark color for price
            hovertemplate=(
                "<b>Day</b>: %{customdata[0]}<br>"
                "<b>Time</b>: %{customdata[1]}<br>"
                "<b>Price</b>: %{y:.4f}<extra></extra>"
            ),
            customdata=[[day, time_str] for time_str in time_strings],
            connectgaps=False,
            showlegend=False # Hide individual day price lines from legend
        ), secondary_y=False)

    # Define colors for features
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5']

    plotted_legend_items = set() # To show legend only once per feature

    # Plot Normalized Features (Secondary Y-axis) - Toggleable, disconnected per day
    for i, original_feature_name in enumerate(feature_columns):
        norm_feature = normalized_feature_columns[original_feature_name] # Get normalized column name
        color = colors[i % len(colors)]

        # Plot each day separately for this normalized feature
        for day, df_day in data_df.groupby("Day"):
            feature_mask = df_day[norm_feature].notna()
            feature_data = df_day.loc[feature_mask, norm_feature]

            if len(feature_data) > 0:
                corresponding_times = df_day.loc[feature_mask, "Time"].apply(lambda x: str(x).split()[-1]).tolist()
                corresponding_plot_indices = df_day.loc[feature_mask, "PlotIndex"].tolist()

                # Show legend only for the first trace of each feature group
                show_legend_for_this_trace = (original_feature_name not in plotted_legend_items)
                if show_legend_for_this_trace:
                    plotted_legend_items.add(original_feature_name)

                fig.add_trace(
                    go.Scatter(
                        x=corresponding_plot_indices,
                        y=feature_data.values.tolist(),
                        mode='lines',
                        name=original_feature_name, # Legend uses original name
                        legendgroup=original_feature_name, # Group traces by original name
                        line=dict(color=color, width=1.5),
                        visible='legendonly', # Start hidden
                        hovertemplate=(
                            "<b>Day</b>: %{customdata[0]}<br>"
                            "<b>Time</b>: %{customdata[1]}<br>"
                            f"<b>{original_feature_name} (Norm)</b>: %{{y:.4f}}<br>" # Show normalized value
                            f"<b>{original_feature_name} (Orig)</b>: %{{customdata[2]:.4f}}<br>" # Show original value
                            "<extra></extra>"
                        ),
                        # Include original feature value in customdata for hover
                        customdata=[[day, time_str, orig_val] for day, time_str, orig_val in zip(
                            [day]*len(corresponding_times),
                            corresponding_times,
                            df_day.loc[feature_mask, original_feature_name].tolist() # Get original values
                        )],
                        connectgaps=False,
                        showlegend=show_legend_for_this_trace # Show legend only once
                    ),
                    secondary_y=True # Plot on the secondary axis
                )

    # Plot Price Jump Pairs if provided
    # Plot Price Jump Pairs if provided
    if price_pairs is not None and len(price_pairs) > 0:
        print(f"  Plotting {len(price_pairs)} price jump pairs...")
        print(f"  Data shape: {len(data_df)} rows")
        print(f"  First few pairs: {price_pairs[:3]}")
        
        for pair_idx, (start_idx, end_idx) in enumerate(price_pairs):
            # Check if indices are valid
            if start_idx >= len(data_df) or end_idx >= len(data_df):
                print(f"  Warning: Pair {pair_idx} has invalid indices ({start_idx}, {end_idx}), skipping...")
                continue
            
            try:
                # Get the corresponding plot indices and prices
                start_plot_idx = data_df.iloc[start_idx]['PlotIndex']
                end_plot_idx = data_df.iloc[end_idx]['PlotIndex']
                start_price = data_df.iloc[start_idx]['Price']
                end_price = data_df.iloc[end_idx]['Price']
                start_time = str(data_df.iloc[start_idx]['Time']).split()[-1]
                end_time = str(data_df.iloc[end_idx]['Time']).split()[-1]
                
                print(f"  Pair {pair_idx}: ({start_idx}, {end_idx}) -> Price diff: {abs(end_price - start_price):.4f}")
                
                # Add start point (green circle)
                fig.add_trace(go.Scatter(
                    x=[start_plot_idx],
                    y=[start_price],
                    mode='markers',
                    marker=dict(size=10, color='green', symbol='circle', 
                               line=dict(color='darkgreen', width=2)),
                    name='Jump Start' if pair_idx == 0 else None,  # Legend only once
                    legendgroup='jump_start',
                    showlegend=(pair_idx == 0),
                    hovertemplate=(
                        "<b>Jump Start</b><br>"
                        f"<b>Time</b>: {start_time}<br>"
                        f"<b>Price</b>: {start_price:.4f}<br>"
                        f"<b>Pair</b>: {pair_idx + 1}<extra></extra>"
                    )
                ), secondary_y=False)
                
                # Add end point (orange circle)
                fig.add_trace(go.Scatter(
                    x=[end_plot_idx],
                    y=[end_price],
                    mode='markers',
                    marker=dict(size=10, color='orange', symbol='circle',
                               line=dict(color='darkorange', width=2)),
                    name='Jump End' if pair_idx == 0 else None,  # Legend only once
                    legendgroup='jump_end',
                    showlegend=(pair_idx == 0),
                    hovertemplate=(
                        "<b>Jump End</b><br>"
                        f"<b>Time</b>: {end_time}<br>"
                        f"<b>Price</b>: {end_price:.4f}<br>"
                        f"<b>Pair</b>: {pair_idx + 1}<br>"
                        f"<b>Price Diff</b>: {abs(end_price - start_price):.4f}<extra></extra>"
                    )
                ), secondary_y=False)
                
                # Add connecting line (gray dashed)
                fig.add_trace(go.Scatter(
                    x=[start_plot_idx, end_plot_idx],
                    y=[start_price, end_price],
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ), secondary_y=False)
            except Exception as e:
                print(f"  Error plotting pair {pair_idx}: {e}")
    else:
        print(f"  No price pairs to plot (pairs is None or empty)")
        if price_pairs is not None:
            print(f"  Price pairs length: {len(price_pairs)}")

    num_days = int(data_df["Day"].max()) + 1
    
    # Create tick positions at the start of each day - EXACTLY like combined_time_series_plot
    tick_positions = []
    tick_labels = []
    for day in range(num_days):
        day_data = data_df[data_df["Day"] == day]
        if len(day_data) > 0:
            tick_positions.append(day_data["PlotIndex"].iloc[0])
            tick_labels.append(f"Day {day}")
    
    # Update the title text
    title_text = 'Combined Multi-Day Price Analysis with Features'
    if price_pairs is not None and len(price_pairs) > 0:
        title_text += f' ({len(price_pairs)} Price Jump Pairs)'
    
    # Update layout - EXACTLY like combined_time_series_plot
    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': 'black'}
        },
        hovermode='x unified',
        height=600,
        width=1600,
        showlegend=True,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    # Update axes - REMOVED secondary_y parameter from update_xaxes
    fig.update_xaxes(
        title_text="Time (HH:MM:SS)",
        tickvals=tick_positions,
        ticktext=tick_labels,
        gridcolor='lightgray'
    )
    
    # Update primary y-axis (Price)
    fig.update_yaxes(
        title_text="Price",
        tickformat='.4f', 
        gridcolor='lightgray',
        secondary_y=False
    )
    
    # Update secondary y-axis (Features)
    fig.update_yaxes(
        title_text="Feature Values",
        tickformat='.10f',
        secondary_y=False
    )
    
    # Set y-axis range with padding - EXACTLY like combined_time_series_plot
    y_padding = (max_price - min_price) * 0.1
    fig.update_yaxes(
        range=[min_price - y_padding, max_price + y_padding],
        secondary_y=False
    )
    
    # Add annotation with statistics - EXACTLY like combined_time_series_plot
    stats_text = (
        f'Number of Days: {num_days}<br>'
        f'Features Available: {len(feature_columns)}<br>'
        f'Price Jump Pairs: {len(price_pairs) if price_pairs else 0}<br>'
        f'Initial Price: ${data_df["Price"].iloc[0]:.2f}<br>'
        f'Final Price: ${data_df["Price"].iloc[-1]:.2f}<br>'
        f'Max Price: ${max_price:.2f}<br>'
        f'Min Price: ${min_price:.2f}<br>'
        f'Range: ${price_range:.2f}<br>'
    )
    
    fig.add_annotation(
        text="annotations",
        hovertext=stats_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        xanchor='left', yanchor='top',
        showarrow=False,
        bgcolor="rgba(255, 250, 205, 0.9)",
        bordercolor="black",
        borderwidth=2,
        borderpad=10,
        align="left",
        font=dict(size=11, family="monospace")
    )
    
    fig.write_html("EBX_day69_BB1,2.html")
    fig.show()

if __name__ == "__main__":
    import sys
    sys.path.append('.')  # Adjust path if your analysis script is elsewhere
    
    # Import from your analysis script (change the filename to match yours)
    from filter import get_pairs_for_day
    
    day_to_plot = 104
    
    # Get pairs and data from analysis
    success, pairs, df_pandas = get_pairs_for_day(day_to_plot)
    
    if success:
        print(f"Successfully loaded day {day_to_plot}")
        print(f"Data shape: {len(df_pandas)} rows")
        print(f"Found {len(pairs)} pairs")
        if len(pairs) > 0:
            print(f"First few pairs: {pairs[:3]}")
        
        # Get all available columns
        available_cols = df_pandas.columns.tolist()
        print(f"\nAvailable columns: {[c for c in available_cols if c not in ['Time', 'Price']][:10]}...")
        # Replace line 80 in your current code:
        # Suppose available_cols is your list of all columns
        selected_features = []

        # Generate BB features: BB4–BB6 with 12 lookbacks
        for bb_num in range(4, 7):  # 4, 5, 6
            for t in range(1, 13):  # T1 to T12
                feat = f"BB{bb_num}_T{t}"
                if feat in available_cols:  # only include if column actually exists
                    selected_features.append(feat)

        # Generate PB features: PB6 and PB13 with 12 lookbacks
        for pb_num in [6, 13]:
            for t in range(1, 13):
                feat = f"PB{pb_num}_T{t}"
                if feat in available_cols:
                    selected_features.append(feat)

        
        print(f"Selected {len(selected_features) if selected_features else 'all'} features to plot")
        if selected_features:
            print(f"Features: {selected_features[:5]}{'...' if len(selected_features) > 5 else ''}")
        
        # Convert to cudf for GPU processing
        gpu_df = cudf.from_pandas(df_pandas)
        
        # Plot with pairs and selected features
        analyze_feature(gpu_df, price_pairs=pairs, selected_features=selected_features)
    else:
        print(f"Failed to load day {day_to_plot}")