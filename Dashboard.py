import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Cars Sales Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
        <style>
        body {
            background-color: black;
        }
        </style>
        """,
    unsafe_allow_html=True,
)


def visualization():
    import numpy as np
    import pandas as pd
    import streamlit as st
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    import joblib
    from xgboost import XGBRegressor
    from sklearn.preprocessing import LabelEncoder
    import streamlit as st
    from streamlit_option_menu import option_menu
    import os

    df = pd.read_csv("Cars sales.csv")
    st.sidebar.header("Cars Sales Dashboard")
    st.sidebar.image("Car.jpg")
    st.sidebar.write(
        "This is a simple dashboard to analyze the car sales data :two_hearts:"
    )

    st.markdown(
        """
        <style>
        body {
            background-color: Black;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <style>
            .metric-box {
                border: 2px solid #00468B;
                border-radius: 8px;
                padding: 20px;
                text-align: center;
            }
            .metric-label {
                font-size: 1.2em;
                font-weight: bold;
            }
            .metric-value {
                font-size: 2em;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )
    metrics = {
        "Total Price": df["Selling_Price"].sum(),
        "Km Driven": df["Km_Driven"].sum(),
        "Max Power": df["Max_Power"].max(),
        "Count Brand": df["Brand"].nunique(),
    }

    def abbreviate_number(num):
        suffixes = {1_000_000_000: "B", 1_000_000: "M", 1_000: "K"}
        for key, suffix in suffixes.items():
            if abs(num) >= key:
                return f"{num / key:.2f}{suffix}"
        return str(num)

    A1, A2, A3, A4 = st.columns(4)
    columns = [A1, A2, A3, A4]
    for col, (label, value) in zip(columns, metrics.items()):
        formatted_value = abbreviate_number(value)
        with col:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{formatted_value}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )
    st.write("______________")

    selected_level = st.sidebar.selectbox(
        "Brand", ["None"] + list(df["Brand"].unique())
    )
    selected_paid = st.sidebar.selectbox(
        "Fuel Type", ["None"] + list(df["Fuel_Type"].unique())
    )
    selected_subject = st.sidebar.selectbox(
        "Transmission Type", ["None"] + list(df["Transmission_Type"].unique())
    )
    selected_year = st.sidebar.selectbox("Year", ["None"] + list(df["Year"].unique()))
    selected_year = st.sidebar.selectbox(
        "Owner Type", ["None"] + list(df["Owner_Type"].unique())
    )

    filtered_df = df
    if selected_level != "None":
        filtered_df = filtered_df[filtered_df["Brand"] == selected_level]
    if selected_paid != "None":
        filtered_df = filtered_df[filtered_df["Fuel_Type"] == selected_paid]
    if selected_subject != "None":
        filtered_df = filtered_df[filtered_df["Transmission_Type"] == selected_subject]
    if selected_year != "None":
        filtered_df = filtered_df[filtered_df["Year"] == selected_year]
    if selected_year != "None":
        filtered_df = filtered_df[filtered_df["Owner_Type"] == selected_year]

    st.write("## Cars Sales Data  🚗💸")
    st.write(filtered_df)
    st.write("______________")
    st.write("### Statistical transactions with data  🔢")
    st.write(df.describe().T)
    st.write(df.describe(include="object").T)
    st.write("______________")
    st.write("## Sales Dashboard 📊")

    grouped_data = df.groupby("Year")["Selling_Price"].sum().reset_index()
    fig = px.line(
        grouped_data,
        x="Year",
        y="Selling_Price",
        title="Total Price Over Years",
        labels={"Year": "Year", "Selling_Price": "Total Price"},
        line_shape="spline",
        markers=True,
    )
    fig.update_layout(
        title_font_size=18,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_traces(
        hovertemplate="<b>Year: %{x}</b><br>Total Price: %{y:,.2f}<extra></extra>",
        line=dict(color="red", width=3),
    )

    st.plotly_chart(fig)  # Make sure this is called only once

    total_price_by_brand = df.groupby("Fuel_Type")["Km_Driven"].sum().reset_index()
    total_price_by_brand = total_price_by_brand.sort_values(
        by="Km_Driven", ascending=False
    )
    fig = px.bar(
        total_price_by_brand,
        x="Fuel_Type",
        y="Km_Driven",
        title="Fuel Type Over Km Driven",
        labels={"Fuel_Type": "Fuel Type", "Km_Driven": "Km Driven"},
        text="Km_Driven",
        height=500,
        color="Km_Driven",
        color_discrete_sequence=px.colors.qualitative.Set1,  # Use Set2 color palette
    )
    fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")
    fig.update_layout(
        title_font_size=18,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_traces(
        hovertemplate="<b>Brand: %{x}</b><br>Km Driven: %{y:,.2f}<extra></extra>"
    )
    st.plotly_chart(fig, use_container_width=True)

    f1, f2, f3 = st.columns([2, 0.3, 2])
    with f1:
        fig_fuel = px.pie(
            df,
            names="Transmission_Type",
            values="Selling_Price",  # Change this to the 'Selling_Price' column
            title="Transmission_Type over Selling_Price",
            color_discrete_sequence=px.colors.qualitative.Set1,  # Use Set2 color palette
        )
        fig_fuel.update_traces(
            textinfo="percent+label",
            marker=dict(line=dict(color="black", width=2)),
        )
        fig_fuel.update_layout(
            title_font_size=24,
            legend_title_text="Fuel Type",
            legend=dict(x=0.8, y=0.5),
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_fuel, use_container_width=True)

    with f3:
        # Assuming 'df' is the dataframe you're working with
        transmission_count = df["Transmission_Type"].value_counts()

        # Create a pie chart with plotly
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=transmission_count.index,
                    values=transmission_count.values,
                    textinfo="percent+label",  # Show percentage and label
                    pull=[0.1, 0],  # Slightly pull one wedge for emphasis (optional)
                    marker=dict(
                        colors=["#66C2A5", "#FC8D62"],
                        line=dict(color="black", width=1.5),
                    ),  # Custom colors and black edges
                )
            ]
        )

        # Customize the layout
        fig.update_layout(
            title="Distribution of Transmission Type",
            title_font=dict(size=24, family="Arial"),
            legend_title="Transmission Type",
            legend_title_font=dict(size=14),
            legend_font=dict(size=12),
            template="plotly_white",  # Clean white background
            margin=dict(l=40, r=40, t=40, b=40),  # Add margins to avoid clipping
        )

        # Show the pie chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    df_grouped = (
        df.groupby(["Fuel_Type", "Transmission_Type"])["Selling_Price"]
        .mean()
        .reset_index()
    )

    # Create a bar chart using Plotly
    fig = px.bar(
        df_grouped,
        x="Fuel_Type",  # Set x-axis to 'Fuel_Type'
        y="Selling_Price",  # Set y-axis to the average 'Selling_Price'
        color="Transmission_Type",  # Use 'Transmission_Type' for grouping by color
        barmode="group",  # Group bars by 'Transmission_Type'
        title="Average Selling Price by Fuel Type and Transmission Type",
        labels={
            "Fuel_Type": "Fuel Type",
            "Selling_Price": "Average Selling Price",
            "Transmission_Type": "Transmission Type",
        },
        color_discrete_sequence=px.colors.qualitative.Set1,  # Set color palette
    )

    # Customize the chart's layout
    fig.update_layout(
        title=dict(
            font=dict(
                size=18,
            )
        ),  # Centered title
        xaxis=dict(
            title="Fuel Type",
            tickangle=45,
            title_font=dict(size=14),
            tickfont=dict(size=12),
        ),
        yaxis=dict(title="Average Selling Price", title_font=dict(size=14)),
        legend_title=dict(text="Transmission Type", font=dict(size=12)),
        margin=dict(l=40, r=40, t=60, b=40),  # Add padding
    )

    # Add grid lines for better readability
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=False)

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        df,
        x="Fuel_Type",  # Set x-axis to 'Fuel_Type'
        y="Selling_Price",  # Set y-axis to 'Selling_Price'
        color="Transmission_Type",  # Group by 'Transmission_Type'
        title="Distribution of Car Prices by Fuel Type and Transmission Type",
        labels={
            "Fuel_Type": "Fuel Type",
            "Selling_Price": "Selling Price (Thousands)",
            "Transmission_Type": "Transmission Type",
        },
        color_discrete_sequence=px.colors.qualitative.Set1,  # Set color palette
        width=800,  # Optional: Adjust the width of the chart
        height=600,  # Optional: Adjust the height of the chart
    )

    # Customize the layout for better appearance
    fig.update_layout(
        title=dict(
            font=dict(
                size=18,
            )
        ),  # Center the title
        xaxis=dict(
            title="Fuel Type",
            tickangle=45,
            title_font=dict(size=14),
            tickfont=dict(size=12),
        ),
        yaxis=dict(title="Selling Price (Thousands)", title_font=dict(size=14)),
        legend_title=dict(text="Transmission Type", font=dict(size=12)),
        margin=dict(l=40, r=40, t=60, b=40),  # Add padding around the plot
    )

    # Add grid lines for better readability
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=False)

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Create the scatter plot using Plotly
    fig = px.scatter(
        df,
        x="Km_Driven",  # X-axis: Kilometers Driven
        y="Selling_Price",  # Y-axis: Selling Price
        color="Fuel_Type",  # Group by Fuel Type
        title="Relationship Between Kilometers Driven and Selling Price by Fuel Type",
        labels={
            "Km_Driven": "Kilometers Driven",
            "Selling_Price": "Selling Price",
            "Fuel_Type": "Fuel Type",
        },
        color_discrete_sequence=px.colors.qualitative.Set1,  # Set color palette
        size_max=10,  # Adjust marker size maximum
        hover_data=["Fuel_Type"],  # Show additional info on hover
    )

    # Customize the layout of the chart
    fig.update_layout(
        title=dict(
            font=dict(
                size=18,
            )
        ),  # Center the title
        xaxis=dict(
            title="Kilometers Driven", title_font=dict(size=14), tickfont=dict(size=12)
        ),
        yaxis=dict(title="Selling Price", title_font=dict(size=14)),
        legend_title=dict(text="Fuel Type", font=dict(size=12)),
        margin=dict(l=40, r=40, t=60, b=40),  # Adjust margins for a cleaner layout
    )

    # Update marker and gridline properties
    fig.update_traces(
        marker=dict(
            size=10, line=dict(width=1, color="white")
        ),  # Marker size and border
        mode="markers",  # Keep scatter plot as markers
    )
    fig.update_yaxes(showgrid=True, gridcolor="lightgray", zeroline=False)

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Count the number of cars for each fuel type
    fuel_counts = df["Fuel_Type"].value_counts().reset_index()
    fuel_counts.columns = ["Fuel_Type", "Count"]

    # Create the bar chart using Plotly
    fig = px.bar(
        fuel_counts,
        x="Fuel_Type",
        y="Count",
        title="Distribution of Car Counts by Fuel Type",
        labels={"Fuel_Type": "Fuel Type", "Count": "Number of Cars"},
        color="Fuel_Type",  # Color by fuel type
        text="Count",  # Display the count on the bars
        color_discrete_sequence=px.colors.qualitative.Set1,  # Use Set2 color palette
    )

    # Customize the layout of the chart
    fig.update_layout(
        title=dict(
            font=dict(
                size=18,
            )
        ),  # Center the title
        xaxis=dict(
            title="Fuel Type",
            tickangle=45,
            title_font=dict(size=14),
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Number of Cars",
            title_font=dict(size=14),
            showgrid=True,
            gridcolor="lightgray",
        ),
        margin=dict(l=40, r=40, t=60, b=40),  # Adjust margins for cleaner layout
        showlegend=False,  # No need for legend since colors represent categories directly
    )

    # Add border and gridline properties
    fig.update_traces(
        marker=dict(line=dict(width=1.5, color="black")),  # Add black border to bars
        textfont=dict(size=12, color="black"),  # Customize text font size and color
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Calculate average selling price grouped by the number of seats
    df_grouped = df.groupby("Seats")["Selling_Price"].mean().reset_index()
    df_grouped = df_grouped.sort_values(by="Selling_Price", ascending=False)

    # Create a Plotly bar chart
    fig = px.bar(
        df_grouped,
        x="Seats",
        y="Selling_Price",
        text="Selling_Price",
        color="Selling_Price",
        color_continuous_scale="Blues",
        color_discrete_sequence=px.colors.qualitative.Set1,  # Use Set2 color palette
        title="Average Selling Price by Number of Seats",
    )

    # Update layout and design to match your Matplotlib style
    fig.update_traces(
        texttemplate="%{text:.2f}",  # Show numeric values on bars
        textposition="outside",
        marker_line_color="black",
        marker_line_width=1.5,
    )

    fig.update_layout(
        title_font=dict(size=16, family="Arial"),
        xaxis=dict(
            title="Number of Seats", title_font=dict(size=14), tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="Average Selling Price",
            title_font=dict(size=14),
            gridcolor="lightgrey",
        ),
        margin=dict(l=40, r=40, t=80, b=40),
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Example data for demonstration (replace with your actual dataset)
    # df = pd.read_csv("your_file.csv")

    # Clean the 'Torque' column by extracting numeric values
    def extract_numeric_torque(torque_value):
        import re

        match = re.search(
            r"(\d+)", str(torque_value)
        )  # Extract the first numeric value
        if match:
            return float(match.group(1))  # Convert to float
        return np.nan  # Handle missing or invalid values

    # Apply cleaning to the 'Torque' column
    if "Torque" in df.columns:
        df["Torque"] = df["Torque"].apply(extract_numeric_torque)

    # Calculate the correlation matrix
    corr_matrix = df[
        ["Selling_Price", "Km_Driven", "Year", "Engine", "Max_Power", "Torque"]
    ].corr()

    # Create the heatmap in Streamlit
    st.markdown("#### Correlation Matrix Between Variables")

    # Customize and render the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        linewidths=0.5,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Correlation Coefficient"},
        annot_kws={"size": 12, "weight": "bold", "color": "black"},
    )
    plt.xticks(fontsize=12, rotation=45, ha="right")
    plt.yticks(fontsize=12)
    plt.gca().set_facecolor("black")
    plt.tight_layout()

    # Display the heatmap in Streamlit
    st.pyplot(plt)

    # Assuming 'df' is the dataframe you're working with
    fig = px.scatter(
        df,
        x="Engine",
        y="Selling_Price",
        color="Fuel_Type",
        symbol="Transmission_Type",
        size_max=100,
        title="Relationship Between Engine Size and Selling Price by Fuel Type",
        labels={"Engine": "Engine Size", "Selling_Price": "Selling Price"},
        color_continuous_scale="Set2",  # Optional: Adjust color scale
        width=800,
        height=600,
    )

    # Customize the layout further
    fig.update_layout(
        title={
            "text": "Relationship Between Engine Size and Selling Price by Fuel Type",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.95,
        },
        xaxis_title="Engine Size",
        yaxis_title="Selling Price",
        legend_title="Fuel Type & Transmission",
        legend_title_font_size=12,
        legend_font_size=12,
        template="plotly",
        showlegend=True,
    )

    # Show the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Assuming 'df' is the dataframe you're working with

    # Define a custom color scale with reversed order
    custom_color_scale = [
        [0, "#6a7377"],  # Start with a lighter color
        [1, "#34393c"],  # End with the darker color
    ]

    # Create the bar plot with custom color scale
    fig = px.bar(
        df,
        x="Year",
        y="Selling_Price",
        title="Average Selling Price by Car Year",
        color="Selling_Price",
        color_continuous_scale=custom_color_scale,  # Apply custom color scale
        labels={"Year": "Car Year", "Selling_Price": "Average Selling Price"},
        height=600,
        width=900,
    )

    # Customize the layout further
    fig.update_layout(
        title={
            "text": "Average Selling Price by Car Year",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.95,
            "font": {"size": 18, "weight": "bold"},
        },
        xaxis_title="Car Year",
        yaxis_title="Average Selling Price",
        xaxis=dict(tickangle=45, tickmode="linear"),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(0,0,0,0.1)"
        ),  # Lighter gridlines for better readability
        template="plotly_white",  # Clean white background
        showlegend=False,  # No need for a legend
        margin=dict(l=40, r=40, t=40, b=40),  # Add margins to avoid clipping
    )

    # Show the plot in Streamlit with responsive width
    st.plotly_chart(fig, use_container_width=True)

    # Assuming 'df' is the dataframe you're working with
    # Count the number of cars by 'Brand'
    brand_count = df["Brand"].value_counts().reset_index()
    brand_count.columns = ["Brand", "Number of Cars"]

    # Create a bar plot with plotly
    fig = px.bar(
        brand_count,
        x="Brand",
        y="Number of Cars",
        title="Distribution of Cars by Brand",
        color="Brand",
        color_discrete_sequence=px.colors.sequential.Viridis,  # Using Viridis color palette
        labels={"Brand": "Car Brand", "Number of Cars": "Number of Cars"},
        height=600,
        width=900,
    )

    # Customize the layout
    fig.update_layout(
        title={
            "text": "Distribution of Cars by Brand",
            "x": 0.5,
            "xanchor": "center",
            "y": 0.95,
            "font": {"size": 18, "weight": "bold"},
        },
        xaxis_title="Car Brand",
        yaxis_title="Number of Cars",
        xaxis=dict(tickangle=45, tickmode="linear", tickfont=dict(size=12)),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)", tickfont=dict(size=12)),
        template="plotly_white",  # Clean white background
        showlegend=False,  # No need for a legend
        margin=dict(l=40, r=40, t=40, b=40),  # Add margins to avoid clipping
    )

    # Show the bar plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Define the top 10 brands
    top_10_brands = df["Brand"].value_counts().head(10).index
    df_top_brands = df[df["Brand"].isin(top_10_brands)]

    # Calculate the average selling price for each brand
    avg_price_by_brand = (
        df_top_brands.groupby("Brand")["Selling_Price"].mean().reset_index()
    )

    # Create a bar plot with Plotly
    fig = px.bar(
        avg_price_by_brand,
        x="Brand",
        y="Selling_Price",
        title="Average Selling Price by Top 10 Brands",
        color="Brand",
        color_discrete_sequence=px.colors.sequential.Viridis,  # Using Viridis color palette
        labels={
            "Brand": "Car Brand",
            "Selling_Price": "Average Selling Price (in thousands)",
        },
        height=600,
        width=900,
    )

    # Customize the layout
    fig.update_layout(
        title={
            "text": "Average Selling Price by Top 10 Brands",
            "x": 0.5,  # Center the title
            "xanchor": "center",
            "y": 0.95,  # Position the title closer to the top
            "font": {"size": 18, "weight": "bold"},
        },
        xaxis_title={"text": "Car Brand", "font": {"size": 14, "family": "Arial"}},
        yaxis_title={
            "text": "Average Selling Price (in thousands)",
            "font": {"size": 14, "family": "Arial"},
        },
        xaxis=dict(
            tickangle=45,  # Rotate labels for better readability
            tickmode="linear",
            tickfont=dict(size=12),
        ),
        yaxis=dict(showgrid=True, tickfont=dict(size=12)),
        template="plotly_white",  # Clean white background
        showlegend=False,  # No need for a legend
        margin=dict(l=40, r=40, t=40, b=40),  # Add margins to avoid clipping
    )

    # Show the bar plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Assuming 'df' is the dataframe you're working with
    # Calculate the maximum power for each car brand
    max_power_by_brand = df.groupby("Brand")["Max_Power"].mean().reset_index()

    # Create a bar plot with Plotly
    fig = px.bar(
        max_power_by_brand,
        x="Brand",
        y="Max_Power",
        title="Maximum Power by Brand",
        color="Brand",  # Color the bars based on the car brand
        color_discrete_sequence=px.colors.sequential.Viridis,  # Using Viridis color palette
        labels={"Brand": "Car Brand", "Max_Power": "Maximum Power (bhp)"},
        height=600,
        width=900,
    )

    # Customize the layout
    fig.update_layout(
        title={
            "text": "Maximum Power by Brand",
            "x": 0.5,  # Center the title
            "xanchor": "center",
            "y": 0.95,  # Position the title closer to the top
            "font": {"size": 18, "weight": "bold"},
        },
        xaxis_title={"text": "Car Brand", "font": {"size": 14, "family": "Arial"}},
        yaxis_title={
            "text": "Maximum Power (bhp)",
            "font": {"size": 14, "family": "Arial"},
        },
        xaxis=dict(
            tickangle=45,  # Rotate labels for better readability
            tickmode="linear",
            tickfont=dict(
                size=12,
            ),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",  # Light grid lines
            tickfont=dict(
                size=12,
            ),
        ),
        template="plotly_white",  # Clean white background
        showlegend=False,  # No need for a legend
        margin=dict(l=40, r=40, t=40, b=40),  # Add margins to avoid clipping
    )

    # Show the bar plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.write("______________")
    st.write("## End of Dashboard")
    st.write("______________")
    st.write("## About")
    st.write(
        "#### This dashboard was created by [Eng. Mohamed Nasr](https://www.linkedin.com/in/medoox369)."
    )
    st.write("#### All ways to get to the code, Dashboard and Report: [GitHub](https://github.com/medoox369/Analysis-of-Car-Sales-Data) | [Kaggle](https://www.kaggle.com/code/medoox369/analysis-of-car-sales-data) | [Colab](https://colab.research.google.com/drive/1XytnbEiKLBumHjccO3L8KSjfrrazRBRa?usp=sharing) | [Power BI](https://app.powerbi.com/view?r=eyJrIjoiYjlhOTVkNjQtOTlmYi00YTdmLWE0YzMtNzkwZjU3NWYzYzMxIiwidCI6ImNmNzIyMWNkLTNiYzYtNDEwMS04NzYyLTU0ZjQ0ZjNiYzg5YSIsImMiOjl9&pageName=1dc7b6d5a2a1b423a6cb) | [Excel](https://drive.google.com/file/d/1g3l7SG-JCNLA5gdiPc6aPIFRJhTiINXv/view?usp=sharing) | [Report](https://docs.google.com/document/d/17wfT1_k_espW1u1Y-nqJzWZ5lPKsE-DN/edit?usp=sharing&ouid=116781748036556031868&rtpof=true&sd=true)")
    st.write("______________")

    st.write("## Thank You :smile:")
    st.write("______________")


def report():
    import numpy as np
    import pandas as pd
    import streamlit as st

    st.write("# Report")
    st.write("______________")
    st.write("## Introduction")
    st.write(
        """This project presents a detailed analysis of car sales data, offering valuable insights into the factors that influence car pricing, performance, and consumer preferences. By examining key attributes such as the car's year, mileage, engine capacity, and fuel type, the analysis aims to uncover trends and patterns that affect the automotive market. Understanding how factors like seller type, transmission type, and previous ownership impact car sales is crucial for businesses looking to enhance their strategies. This review also explores how features like brand, model, and additional car details contribute to price variations and buyer decision-making. Through this comprehensive analysis, the project provides actionable recommendations that can help companies navigate the competitive automotive market and optimize their pricing strategies for better consumer engagement and profitability."""
    )
    st.write("______________")
    st.write("## Data Overview")
    st.write("The data used in this analysis includes the following columns:")
    st.write(
        "- Year: The year of manufacture of the car. This column is used to determine the age of the car and its impact on the selling price."
    )
    st.write(
        "- Selling_Price: The price at which the car is being sold. It represents the amount paid for the car."
    )
    st.write(
        "- Km_Driven: The number of kilometers the car has driven. This column indicates the extent of usage and its effect on the car's condition and price."
    )
    st.write(
        "- Fuel_Type: The type of fuel the car uses (e.g., petrol, diesel, electric). Fuel type affects the car's fuel efficiency and price."
    )
    st.write(
        "- Seller_Type: The type of seller (e.g., dealer, individual owner). This column helps identify whether the car is being sold by a dealer or a previous owner."
    )
    st.write(
        "- Transmission_Type: The type of transmission (manual or automatic). It impacts driving comfort and buyer preferences."
    )
    st.write(
        "- Owner_Type: The number of previous owners (e.g., first owner, second owner). This column helps assess the car’s condition based on ownership history."
    )
    st.write(
        "- Mileage: The distance the car travels per liter of fuel (km/l). This column is used to evaluate the car's fuel efficiency."
    )
    st.write(
        "- Engine: The engine capacity of the car (in liters). This column affects the car's performance and fuel consumption."
    )
    st.write(
        "- Max_Power: The maximum power output of the car's engine (in horsepower). This column determines the car’s performance and power."
    )
    st.write(
        "- Torque: The maximum torque generated by the engine (in Newton meters). Torque affects the car's acceleration and overall performance."
    )
    st.write(
        "- Seats: The number of seats in the car. This column helps assess the size of the car and its suitability for families or commercial use."
    )
    st.write(
        "- Brand: The brand of the car (e.g., Toyota, Mercedes). The brand can influence the car's value and price."
    )
    st.write(
        "- Model: The specific model of the car (e.g., Corolla, Camry). This column helps to determine the exact model and estimate its value."
    )
    st.write(
        "- Details: Additional details about the car, such as color, special features, or service history. This column provides extra information to help evaluate the car's condition."
    )
    st.write("______________")
    st.write("## Key Findings")
    st.write(
        "- **Steady Growth:** The total price experienced a gradual increase from 1995 until around 2010, indicating a consistent upward trend."
    )
    st.write(
        "- **Rapid Growth:** A substantial surge in total price was observed between 2010 and 2015, suggesting a potential shift in market dynamics or successful marketing campaigns."
    )
    st.write(
        "- **Peak in 2015:** The total price reached its peak in 2015, indicating a saturation point or a change in market conditions."
    )
    st.write(
        "- **Subsequent Decline:** Following the peak in 2015, the total price experienced a decline, possibly due to factors such as increased competition, economic downturns, or changes in consumer preferences."
    )
    st.write(
        "- **Diesel Dominance:** Diesel-powered vehicles have significantly outperformed other fuel types in terms of total kilometers driven, indicating a strong preference for diesel engines among consumers."
    )
    st.write(
        "- **Petrol Popularity:** Petrol-powered vehicles occupy the second position, suggesting a substantial market share for petrol-driven cars."
    )
    st.write(
        "- **Niche Segments:** CNG and LPG-powered vehicles have a significantly lower market share compared to diesel and petrol, indicating that they cater to niche markets or specific regional preferences."
    )
    st.write(
        "- **Manual Transmission Dominance:** Manual transmissions account for a significant majority of the vehicles, representing '78.54%' of the total."
    )
    st.write(
        "- **Automatic Transmission Minority:** Automatic transmissions make up a smaller portion of the market, accounting for '21.5%' of the total."
    )
    st.write(
        "- **Automatic Premium:** Vehicles equipped with automatic transmissions generally command higher average selling prices compared to their manual counterparts, regardless of the fuel type. This suggests that consumers are willing to pay a premium for the convenience and comfort associated with automatic transmissions."
    )
    st.write(
        "- **Diesel Premium:** Diesel-powered vehicles, especially those with automatic transmissions, tend to have the highest average selling prices. This indicates that diesel engines are often associated with higher performance and durability, justifying the higher price tag."
    )
    st.write(
        "- **CNG and LPG Discounts:** Vehicles powered by CNG and LPG have the lowest average selling prices, reflecting their reputation as more affordable and environmentally friendly options."
    )
    st.write(
        "- **Dominance of Diesel and Petrol:** Diesel and petrol-powered vehicles significantly outnumber vehicles using CNG and LPG. This suggests a strong preference for traditional fuel types among consumers."
    )
    st.write(
        "- **Diesel Popularity:** Diesel vehicles have the highest count, indicating a strong market share for diesel-powered cars. This could be attributed to factors such as diesel engines' reputation for fuel efficiency and torque."
    )
    st.write(
        "- **Niche Market for CNG and LPG:** CNG and LPG-powered vehicles constitute a very small portion of the market, suggesting that they cater to a niche market of environmentally conscious consumers or those looking for more affordable options."
    )
    st.write(
        "- **Positive Correlation:** There is a generally positive correlation between the number of seats and the average selling price. As the number of seats increases, the average selling price tends to increase as well."
    )
    st.write(
        "- **Price Jump at 8 Seats:** There is a significant jump in the average selling price when moving from vehicles with 7 seats to those with 8 seats. This suggests that vehicles with 8 seats, often categorized as larger SUVs or vans, command a higher premium in the market."
    )
    st.write(
        "- **Price Decrease at 14 Seats:** The average selling price drops significantly for vehicles with 14 seats. This could be due to a smaller market segment for such large vehicles or the presence of more specialized, lower-cost options in this category."
    )
    st.write(
        "- **Dominance of Maruti and Hyundai:** Maruti and Hyundai brands have the highest number of cars, indicating a strong market share for these brands. This could be attributed to factors such as affordability, reliability, and extensive dealership networks."
    )
    st.write(
        "- **Tiered Market:** The data reveals a tiered market with a few dominant brands at the top and a long tail of smaller brands with significantly fewer cars. This suggests that the automotive market is highly concentrated with a few major players."
    )
    st.write(
        "- **Niche Brands:** While Maruti and Hyundai dominate, there is a presence of several other brands, indicating a diverse range of options for consumers. These niche brands may cater to specific consumer segments or offer unique features and qualities."
    )
    st.write(
        "- **Price Variation:** There is a significant variation in the average selling prices among the top 10 car brands. This indicates that factors such as brand reputation, vehicle features, and target market play a crucial role in determining the price of a car."
    )
    st.write(
        "- **Premium Brands:** Brands like Volkswagen and Toyota command higher average selling prices, suggesting that they are positioned as premium brands with higher quality and features."
    )
    st.write(
        "- **Value Brands:** On the other hand, brands like Chevrolet and Ford have lower average selling prices, indicating that they are positioned as more affordable options."
    )
    st.write(
        "- **Market Segmentation:** The data suggests that the automotive market is segmented based on price, with different brands catering to different customer segments."
    )
    st.write(
        "- **Wide Range of Power Outputs:** There is a significant variation in the maximum power output among the different car brands. This indicates that factors such as engine size, vehicle type, and target market play a crucial role in determining the power of a car."
    )
    st.write(
        "- **High-Performance Brands:** Some brands, such as [Volvo] have significantly higher maximum power outputs, suggesting that they are positioned as performance-oriented brands."
    )
    st.write(
        "- **Efficiency-Focused Brands:** On the other hand, brands with lower maximum power outputs may be focusing on fuel efficiency and lower emissions."
    )
    st.write(
        "- **Market Segmentation:** The data suggests that the automotive market is segmented based on power output, with different brands catering to different customer segments."
    )
    st.write("______________")
    st.write("## Potential Implications and Recommendations")
    st.write(
        "- **Market Trends**: The data reveals a strong inclination towards diesel and petrol engines, suggesting a preference for conventional fuel types."
    )
    st.write(
        "- **Product Innovations**: Manufacturers can focus on enhancing diesel and petrol-powered vehicles with better fuel efficiency and reduced emissions to meet market demands."
    )
    st.write(
        "- **Infrastructure Needs**: The dominance of diesel and petrol highlights the necessity for robust fuel station networks to support these vehicles effectively."
    )
    st.write(
        "- **Government Policies**: Regulations on emissions and fuel subsidies play a significant role in shaping consumer preferences and market trends."
    )
    st.write(
        "- **Environmental Impact**: The widespread use of conventional fuels raises concerns about emissions, encouraging a shift toward alternative fuels and cleaner technologies."
    )
    st.write(
        "- **Market Segmentation**: Automakers and dealers can target specific segments by offering features like automatic transmissions and luxury options tailored to premium customers."
    )
    st.write(
        "- **Pricing Strategy**: By analyzing factors like fuel type, transmission, and seating capacity, companies can optimize pricing to align with consumer preferences."
    )
    st.write(
        "- **Inventory Management**: Dealers should adjust inventory based on trends in transmission and fuel preferences to meet market demands effectively."
    )
    st.write(
        "- **Future Trends**: Monitoring shifts in transmission and fuel preferences enables businesses to anticipate market demands and innovate accordingly."
    )
    st.write(
        "- **Brand Strategy**: Market analysis helps car manufacturers refine their brand positioning, ensuring competitiveness by highlighting unique value propositions."
    )
    st.write(
        "- **Consumer Insights**: Understanding buyer preferences for features, fuel types, and performance helps manufacturers and dealers tailor their offerings."
    )
    st.write(
        "- **Performance Focus**: High-power engines appeal to performance enthusiasts, while fuel-efficient models attract budget-conscious consumers."
    )
    st.write(
        "- **Product Development**: Automakers can design vehicles with features that align with specific customer needs, such as seating configurations or advanced technology."
    )
    st.write(
        "- **Marketing Strategy**: Tailored campaigns can target customer segments effectively, emphasizing either performance or affordability based on market data."
    )
    st.write(
        "- **Economic Factors**: Fluctuations in the economy directly influence consumer spending, affecting pricing strategies and market dynamics."
    )
    st.write("______________")
    st.write("### Conclusion of the Report")
    st.write(
        """
    Conclusion
    To remain competitive in a rapidly changing automotive market, companies must leverage the power of data analytics, focus on brand differentiation, and embrace innovation. Understanding customer behavior and preferences is key to targeting the right audience with the right product offerings. By implementing predictive analytics and expanding their focus on sustainable vehicle solutions, companies can secure their place in the future of the automotive industry.

    The key to long-term success lies in staying informed and agile, using data to drive decisions, and continuously innovating to meet consumer expectations. Companies that follow these steps will not only advance in the labor market but also position themselves as leaders in a highly competitive landscape."""
    )
    st.write("______________")
    st.write("## End of Report")
    st.write("______________")
    st.write("## About")
    st.write(
        "#### This dashboard was created by [Eng. Mohamed Nasr](https://www.linkedin.com/in/medoox369)."
    )
    st.write("#### All ways to get to the code, Dashboard and Report: [GitHub](https://github.com/medoox369/Analysis-of-Car-Sales-Data) | [Kaggle](https://www.kaggle.com/code/medoox369/analysis-of-car-sales-data) | [Colab](https://colab.research.google.com/drive/1XytnbEiKLBumHjccO3L8KSjfrrazRBRa?usp=sharing) | [Power BI](https://app.powerbi.com/view?r=eyJrIjoiYjlhOTVkNjQtOTlmYi00YTdmLWE0YzMtNzkwZjU3NWYzYzMxIiwidCI6ImNmNzIyMWNkLTNiYzYtNDEwMS04NzYyLTU0ZjQ0ZjNiYzg5YSIsImMiOjl9&pageName=1dc7b6d5a2a1b423a6cb) | [Excel](https://drive.google.com/file/d/1g3l7SG-JCNLA5gdiPc6aPIFRJhTiINXv/view?usp=sharing) | [Report](https://docs.google.com/document/d/17wfT1_k_espW1u1Y-nqJzWZ5lPKsE-DN/edit?usp=sharing&ouid=116781748036556031868&rtpof=true&sd=true)")
    st.write("______________")
    st.write("## Thank You :smile:")


import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler


def Predictions():
    # 🚗 Load pre-trained model and transformers
    categorical_cols = [
        "Fuel_Type",
        "Seller_Type",
        "Transmission_Type",
        "Owner_Type",
        "Brand",
        "Model",
        "Torque",
    ]
    numerical_cols = ["Year", "Mileage", "Engine", "Max_Power"]

    # Load label encoders for categorical features
    label_encoders = {
        col: joblib.load(f"label_encoder_{col.lower()}.pkl") for col in categorical_cols
    }

    # Load trained machine learning model, scaler, and feature selector
    model = joblib.load("best_model_xgboost.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_selector = joblib.load("feature_selector.pkl")

    # 🛠️ Function to encode new data using saved label encoders
    def encode_new_data(new_data, label_encoders):
        encoded_data = new_data.copy()
        for col, le in label_encoders.items():
            if encoded_data[col].dtype == "object":
                # ⚠️ Handle unseen categories by assigning the first class
                for idx, val in enumerate(encoded_data[col]):
                    if val not in le.classes_ and val != "None":  # Skip 'None' values
                        encoded_data.loc[idx, col] = le.classes_[0]
                encoded_data[col] = le.transform(encoded_data[col].astype(str))
        return encoded_data

    # 🌟 Streamlit Interface: Display input fields for the user to fill in car details
    st.title("Car Price Predictor 🚗💸")

    # Add "None" as the first option for each dropdown list
    brand = st.selectbox(
        "🔹 Select Brand",
        ["None"]
        + [
            "Maruti",
            "Skoda",
            "Honda",
            "Hyundai",
            "Toyota",
            "Ford",
            "Renault",
            "Mahindra",
            "Tata",
            "Chevrolet",
            "Datsun",
            "Jeep",
            "Mercedes-Benz",
            "Mitsubishi",
            "Audi",
            "Volkswagen",
            "BMW",
            "Nissan",
            "Lexus",
            "Jaguar",
            "Land",
            "MG",
            "Volvo",
            "Daewoo",
            "Kia",
            "Fiat",
            "Force",
            "Ambassador",
            "Ashok",
            "Isuzu",
            "Opel",
        ],
        key="brand_select",
    )

    model_name = st.selectbox(
        "🔹 Select Model",
        ["None"]
        + [
            "Swift",
            "Rapid",
            "City",
            "i20",
            "Xcent",
            "Wagon",
            "800",
            "Etios",
            "Figo",
            "Duster",
            "Zen",
            "KUV",
            "Ertiga",
            "Alto",
            "Verito",
            "WR-V",
            "SX4",
            "Tigor",
            "Baleno",
            "Enjoy",
            "Omni",
            "Vitara",
            "Verna",
            "GO",
            "Safari",
            "Compass",
            "Fortuner",
            "Innova",
            "B",
            "Amaze",
            "Pajero",
            "Ciaz",
            "Jazz",
            "A6",
            "Corolla",
            "New",
            "Manza",
            "i10",
            "Ameo",
            "Vento",
            "EcoSport",
            "X1",
            "Celerio",
            "Polo",
            "Eeco",
            "Scorpio",
            "Freestyle",
            "Passat",
            "Indica",
            "XUV500",
            "Indigo",
            "Terrano",
            "Creta",
            "KWID",
            "Santro",
            "Q5",
            "ES",
            "XF",
            "Wrangler",
            "Rover",
            "S-Class",
            "5",
            "X4",
            "Superb",
            "E-Class",
            "Hector",
            "XC40",
            "Q7",
            "Elantra",
            "XE",
            "Nexon",
            "CLA",
            "Glanza",
            "3",
            "Camry",
            "XC90",
            "Ritz",
            "Grand",
            "Matiz",
            "Zest",
            "Getz",
            "Elite",
            "Brio",
            "Hexa",
            "Sunny",
            "Micra",
            "Ssangyong",
            "Quanto",
            "Accent",
            "Ignis",
            "Marazzo",
            "Tiago",
            "Thar",
            "Sumo",
            "Bolero",
            "GL-Class",
            "Beat",
            "A-Star",
            "XUV300",
            "Nano",
            "GTI",
            "V40",
            "CR-V",
            "EON",
            "RediGO",
            "Captiva",
            "Fiesta",
            "Seltos",
            "Civic",
            "Sail",
            "Venture",
            "Classic",
            "BR-V",
            "Ecosport",
            "Aria",
            "TUV",
            "Bolt",
            "Accord",
            "Xylo",
            "Grande",
            "S-Cross",
            "Yaris",
            "Tavera",
            "Linea",
            "Endeavour",
            "Aveo",
            "Triber",
            "Fusion",
            "Octavia",
            "A4",
            "XL6",
            "Santa",
            "Spark",
            "Aspire",
            "Optra",
            "Mobilio",
            "BRV",
            "X6",
            "Cruze",
            "GLA",
            "6",
            "NuvoSport",
            "Scala",
            "Lodgy",
            "Pulse",
            "Supro",
            "Sonata",
            "Renault",
            "Kicks",
            "Jetta",
            "M-Class",
            "Teana",
            "Yeti",
            "Q3",
            "Gurkha",
            "Logan",
            "A3",
            "Dzire",
            "Ikon",
            "Fluence",
            "Xenon",
            "One",
            "7",
            "S60",
            "Lancer",
            "X7",
            "Fabia",
            "Platinum",
            "Captur",
            "Gypsy",
            "Koleos",
            "CLASSIC",
            "Harrier",
            "Punto",
            "Avventura",
            "Laura",
            "Leyland",
            "MUX",
            "Astra",
            "Tucson",
            "Esteem",
            "Winger",
            "Qualis",
            "Spacio",
            "Venue",
            "CrossPolo",
            "Kodiaq",
            "D-Max",
            "X3",
            "Land",
            "X5",
            "Trailblazer",
            "MU",
            "GLC",
            "XC60",
            "S90",
            "S-Presso",
        ],
        key="model_select",
    )

    year = st.slider("🔹 Select Year", 1994, 2025, key="year_slider")
    mileage = st.number_input("🔹 Mileage (in km/l)", min_value=0, key="mileage_input")
    engine = st.number_input(
        "🔹 Engine Size (in cc)", min_value=0.0, key="engine_input"
    )
    max_power = st.number_input(
        "🔹 Max Power (in bhp)", min_value=0.0, key="power_input"
    )
    torque = st.number_input("🔹 Torque (in Nm)", min_value=0.0, key="torque_input")

    # Updated Fuel Type with 4 options
    fuel_type = st.selectbox(
        "🔹 Fuel Type",
        ["None", "Petrol", "Diesel", "CNG", "Electric"],
        key="fuel_select",
    )

    seller_type = st.selectbox(
        "🔹 Seller Type",
        ["None", "Individual", "Dealer", "Trustmark Dealer"],
        key="seller_select",
    )
    transmission_type = st.selectbox(
        "🔹 Transmission Type",
        ["None", "Manual", "Automatic"],
        key="transmission_select",
    )

    # Updated Owner Type with 5 options
    owner_type = st.selectbox(
        "🔹 Owner Type",
        [
            "None",
            "First Owner",
            "Second Owner",
            "Third Owner",
            "Fourth & Above Owner",
            "Test Drive Car",
        ],
        key="owner_select",
    )

    # 🏁 When the "Predict Price" button is clicked, collect data, preprocess it, and make predictions
    if st.button("🔮 Predict Price", key="predict_button"):
        # 📝 Create a DataFrame for the input data
        new_car_data = pd.DataFrame(
            {
                "Brand": [brand if brand != "None" else None],
                "Model": [model_name if model_name != "None" else None],
                "Year": [year],
                "Mileage": [mileage],
                "Engine": [engine],
                "Max_Power": [max_power],
                "Torque": [
                    f"{torque}Nm@ 4000rpm"
                ],  # 🚗 Display torque in the desired format
                "Fuel_Type": [fuel_type if fuel_type != "None" else None],
                "Seller_Type": [seller_type if seller_type != "None" else None],
                "Transmission_Type": [
                    transmission_type if transmission_type != "None" else None
                ],
                "Owner_Type": [owner_type if owner_type != "None" else None],
            }
        )

        # Remove rows where "None" is selected for required fields (e.g., 'Brand', 'Model')
        if new_car_data[["Brand", "Model"]].isnull().any().any():
            st.error(
                "Please select valid options for all required fields (Brand, Model)."
            )
        else:
            # Encode the new data
            new_car_data_encoded = encode_new_data(new_car_data, label_encoders)

            # Standardize numerical features using StandardScaler
            new_car_data_encoded[numerical_cols] = scaler.transform(
                new_car_data_encoded[numerical_cols]
            )

            # Apply feature selection
            input_features = feature_selector.transform(new_car_data_encoded)

            # Make prediction
            prediction = model.predict(input_features)
            st.success(f"🎉 Predicted Price: ${prediction[0]:,.2f}")

    st.write("______________")
    st.write("## End of Predictor")
    st.write("______________")
    st.write("## About")
    st.write(
        "#### This dashboard was created by [Eng. Mohamed Nasr](https://www.linkedin.com/in/medoox369)."
    )
    "#### All ways to get to the code, Dashboard and Report: [GitHub](https://github.com/medoox369/Analysis-of-Car-Sales-Data) | [Kaggle](https://www.kaggle.com/code/medoox369/analysis-of-car-sales-data) | [Colab](https://colab.research.google.com/drive/1XytnbEiKLBumHjccO3L8KSjfrrazRBRa?usp=sharing) | [Power BI](https://app.powerbi.com/view?r=eyJrIjoiYjlhOTVkNjQtOTlmYi00YTdmLWE0YzMtNzkwZjU3NWYzYzMxIiwidCI6ImNmNzIyMWNkLTNiYzYtNDEwMS04NzYyLTU0ZjQ0ZjNiYzg5YSIsImMiOjl9&pageName=1dc7b6d5a2a1b423a6cb) | [Excel](https://drive.google.com/file/d/1g3l7SG-JCNLA5gdiPc6aPIFRJhTiINXv/view?usp=sharing) | [Report](https://docs.google.com/document/d/17wfT1_k_espW1u1Y-nqJzWZ5lPKsE-DN/edit?usp=sharing&ouid=116781748036556031868&rtpof=true&sd=true)"
    st.write("______________")
    st.write("## Thank You :smile:")

def contact():
    import streamlit as st
    import joblib
    from streamlit_option_menu import option_menu
    import os
    st.write("# Contact")
    st.write("______________")
    st.write("## About")
    st.write(
        "#### This dashboard was created by [Eng. Mohamed Nasr](https://www.linkedin.com/in/medoox369)."
    )
    st.write("#### The data used in this dashboard is from Dr. Mustafa Othman.")
    st.write(
        "#### The code for this dashboard is available on [GitHub](https://github.com/medoox369/Analysis-of-Car-Sales-Data)"
    )
    st.write("______________")
    st.write(
        "#### I would like to express my heartfelt gratitude to Dr. Mostafa Osman for all the effort and dedication you have shown during this data science training course. Your guidance has greatly contributed to shaping my academic and professional future and has prepared me to excel in the job market with confidence. It is truly an honor to be one of your students. Your deep knowledge and exceptional teaching methods have been a great source of inspiration to me. I have gained invaluable skills from you, not only technically but also personally, through the values of commitment, hard work, and creativity. I am incredibly proud and thankful to have had the privilege of being your student."
    )
    st.write(
        "#### For those wishing to connect with Dr. Mustafa Othman, here are his contact details:"
    )
    st.write(
        "#### [WhatsApp](https://wa.me/+201066709959) | [LinkedIn](https://www.linkedin.com/in/mustafaelnahas) | [YouTube](https://www.youtube.com/@MustafaOthman) | [Udemy](https://www.udemy.com/user/mustafa-othman-3/)"
    )
    st.write(
        "#### Thank you again, Dr. Mustafa, for your tireless efforts and impactful teaching. I am grateful for the opportunity to learn from you and look forward to applying the knowledge and skills you have imparted to me in my future endeavors. I wish you continued success and fulfillment in all your endeavors. Thank you for being an exceptional teacher and mentor. I am truly grateful for your guidance and support. May God bless you. Thank you."
    )
    st.write("______________")
    st.write(
        "### For inquiries or more information about me, contact me at: [WhatsApp](https://wa.me/+201276977748) | [LinkedIn](https://www.linkedin.com/in/medoox369) | [GitHub](https://github.com/medoox369) | [Email](mailto:https://medoox369@gmail.com) | [Kaggle](https://www.kaggle.com/medoox369)"
    )
    st.write("______________")
    st.write(
        "#### All ways to get to the code, Dashboard and Report: [GitHub](https://github.com/medoox369/Analysis-of-Car-Sales-Data) | [Kaggle](https://www.kaggle.com/code/medoox369/analysis-of-car-sales-data) | [Colab](https://colab.research.google.com/drive/1XytnbEiKLBumHjccO3L8KSjfrrazRBRa?usp=sharing) | [Power BI](https://app.powerbi.com/view?r=eyJrIjoiYjlhOTVkNjQtOTlmYi00YTdmLWE0YzMtNzkwZjU3NWYzYzMxIiwidCI6ImNmNzIyMWNkLTNiYzYtNDEwMS04NzYyLTU0ZjQ0ZjNiYzg5YSIsImMiOjl9&pageName=1dc7b6d5a2a1b423a6cb) | [Excel](https://drive.google.com/file/d/1g3l7SG-JCNLA5gdiPc6aPIFRJhTiINXv/view?usp=sharing) | [Report](https://docs.google.com/document/d/17wfT1_k_espW1u1Y-nqJzWZ5lPKsE-DN/edit?usp=sharing&ouid=116781748036556031868&rtpof=true&sd=true)"
    )
    st.write("______________")
    st.write("## End of Project")
    st.write("______________")
    st.write("## Thank You :smile:")


def streamlit_menu():
    selected = option_menu(
        menu_title=None,
        options=["Visualization", "Report", "Predictions", "Contact"],
        icons=["bar-chart-line-fill", "book", "graph-up-arrow", "envelope"],
        menu_icon="cast",
        orientation="horizontal",
    )
    return selected


# Initialize session state
if "selected" not in st.session_state:
    st.session_state["selected"] = None

selected = streamlit_menu()

if selected != st.session_state["selected"]:
    st.session_state["selected"] = selected
if st.session_state["selected"] == "Visualization":
    visualization()
elif st.session_state["selected"] == "Report":
    report()
elif st.session_state["selected"] == "Predictions":
    Predictions()
elif st.session_state["selected"] == "Contact":
    contact()
