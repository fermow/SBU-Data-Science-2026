import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
pd.set_option("display.max_columns", None)
sb.set_style("whitegrid")

df_2015 = pd.read_csv("../data/exploratory_data_analysis_of_airline_delays_and_cancellations/2015.csv", encoding='ISO-8859-1')
df_2016 = pd.read_csv("../data/exploratory_data_analysis_of_airline_delays_and_cancellations/2016.csv", encoding='ISO-8859-1')
df_2017 = pd.read_csv("../data/exploratory_data_analysis_of_airline_delays_and_cancellations/2017.csv", encoding='ISO-8859-1')

df = pd.concat([df_2015, df_2016, df_2017], ignore_index=True)

missing_counts = df.isnull().sum()
missing_percentages = df.isnull().mean() * 100
missing_summary = pd.DataFrame({"missing_count": missing_counts, "missing_percentage": missing_percentages})
missing_summary = missing_summary.sort_values(by="missing_percentage", ascending=False)

missing_plot = missing_summary[missing_summary["missing_percentage"] > 0].sort_values(by="missing_percentage", ascending=False)
plt.figure(figsize=(12, 6))
sb.barplot(x=missing_plot.index, y=missing_plot["missing_percentage"])
plt.xticks(rotation=90)
plt.ylabel("Missing Percentage")
plt.xlabel("Columns")
plt.title("Percentage of Missing Values by Column")
plt.tight_layout()
plt.savefig("figures/missing_values.png", dpi=300, bbox_inches="tight")
plt.close()

df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
df["MONTH"] = df["FL_DATE"].dt.month
df["YEAR"] = df["FL_DATE"].dt.year
df["DAY_OF_WEEK"] = df["FL_DATE"].dt.day_name()
df["DEP_HOUR"] = df["CRS_DEP_TIME"] // 100

def get_time_of_day(hour):
    if hour < 5: return "Late Night"
    elif hour < 8: return "Early Morning"
    elif hour < 12: return "Morning"
    elif hour < 18: return "Afternoon"
    else: return "Evening"

df["TIME_OF_DAY"] = df["DEP_HOUR"].apply(get_time_of_day)

df.groupby("YEAR").size().plot(kind="bar")
plt.title("Number of Flights per Year")
plt.xlabel("Year")
plt.ylabel("Flights")
plt.tight_layout()
plt.savefig("figures/yearly_flights.png", dpi=300, bbox_inches="tight")
plt.close()

df.groupby("MONTH").size().plot(kind="bar")
plt.title("Number of Flights per Month")
plt.xlabel("Month")
plt.ylabel("Flights")
plt.tight_layout()
plt.savefig("figures/monthly_flights.png", dpi=300, bbox_inches="tight")
plt.close()

sb.histplot(df["ARR_DELAY"].dropna(), bins=200)
plt.title("Distribution of Arrival Delay")
plt.xlim(-100, 200)
plt.tight_layout()
plt.savefig("figures/arr_delay_hist.png", dpi=300, bbox_inches="tight")
plt.close()

sb.histplot(df["DEP_DELAY"].dropna(), bins=500)
plt.title("Distribution of Departure Delay")
plt.xlim(-50, 200)
plt.tight_layout()
plt.savefig("figures/dep_delay_hist.png", dpi=300, bbox_inches="tight")
plt.close()

sb.histplot(df["DISTANCE"].dropna(), bins=100)
plt.title("Distribution of Flight Distance")
plt.tight_layout()
plt.savefig("figures/distance_hist.png", dpi=300, bbox_inches="tight")
plt.close()

top_carriers = df["OP_CARRIER"].value_counts().head(10)
plt.figure(figsize=(10, 6))
sb.barplot(x=top_carriers.index, y=top_carriers.values)
plt.title("Top 10 Carriers by Number of Flights")
plt.xlabel("Carrier")
plt.ylabel("Number of Flights")
plt.tight_layout()
plt.savefig("figures/carrier_counts.png", dpi=300, bbox_inches="tight")
plt.close()

top_origins = df["ORIGIN"].value_counts().head(10)
plt.figure(figsize=(12, 6))
sb.barplot(x=top_origins.index, y=top_origins.values)
plt.title("Top Origin Airports")
plt.xlabel("Airport")
plt.ylabel("Departures")
plt.tight_layout()
plt.savefig("figures/airport_counts.png", dpi=300, bbox_inches="tight")
plt.close()

monthly_delay = df.groupby("MONTH")["ARR_DELAY"].mean()
plt.figure(figsize=(10, 5))
sb.lineplot(x=monthly_delay.index, y=monthly_delay.values, marker="o")
plt.title("Average Arrival Delay by Month")
plt.xlabel("Month")
plt.ylabel("Average Arrival Delay")
plt.tight_layout()
plt.savefig("figures/monthly_delay.png", dpi=300, bbox_inches="tight")
plt.close()

yearly_delay = df.groupby("YEAR")["ARR_DELAY"].mean()
plt.figure(figsize=(10, 5))
sb.lineplot(x=yearly_delay.index, y=yearly_delay.values, marker="o")
plt.title("Average Arrival Delay by Year")
plt.xlabel("Year")
plt.ylabel("Average Arrival Delay")
plt.tight_layout()
plt.savefig("figures/yearly_delay.png", dpi=300, bbox_inches="tight")
plt.close()

cancel_rate_year = df.groupby("YEAR")["CANCELLED"].mean() * 100
plt.figure(figsize=(10, 5))
sb.lineplot(x=cancel_rate_year.index, y=cancel_rate_year.values, marker="o")
plt.title("Cancellation Rate by Year")
plt.xlabel("Year")
plt.ylabel("Cancellation Rate (%)")
plt.tight_layout()
plt.savefig("figures/cancellation_rate.png", dpi=300, bbox_inches="tight")
plt.close()

yearly_analysis = df.groupby("YEAR").agg({"ARR_DELAY": "mean", "CANCELLED": "mean"})
yearly_analysis["CANCELLED"] *= 100
plt.figure(figsize=(8, 6))
sb.scatterplot(data=yearly_analysis, x="ARR_DELAY", y="CANCELLED", s=120)
for year in yearly_analysis.index:
    plt.text(yearly_analysis.loc[year, "ARR_DELAY"], yearly_analysis.loc[year, "CANCELLED"], str(year))
plt.title("Relationship Between Arrival Delay and Cancellation Rate")
plt.xlabel("Average Arrival Delay")
plt.ylabel("Cancellation Rate (%)")
plt.tight_layout()
plt.savefig("figures/delay_cancel_scatter.png", dpi=300, bbox_inches="tight")
plt.close()

day_delay = df.groupby("DAY_OF_WEEK")["ARR_DELAY"].mean()
plt.figure(figsize=(10, 5))
sb.barplot(x=day_delay.index, y=day_delay.values)
plt.title("Average Arrival Delay by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Arrival Delay")
plt.tight_layout()
plt.savefig("figures/dow_delay.png", dpi=300, bbox_inches="tight")
plt.close()

hourly_delay = df.groupby("DEP_HOUR")["ARR_DELAY"].mean()
plt.figure(figsize=(12, 5))
sb.lineplot(x=hourly_delay.index, y=hourly_delay.values, marker="o")
plt.xticks(range(24))
plt.title("Average Arrival Delay by Departure Hour")
plt.xlabel("Departure Hour")
plt.ylabel("Average Arrival Delay")
plt.tight_layout()
plt.savefig("figures/hourly_delay.png", dpi=300, bbox_inches="tight")
plt.close()

time_of_day_delay = df.groupby("TIME_OF_DAY")["ARR_DELAY"].mean()
plt.figure(figsize=(10, 5))
sb.barplot(x=time_of_day_delay.index, y=time_of_day_delay.values)
plt.title("Average Arrival Delay by Time of Day")
plt.xlabel("Time of Day")
plt.ylabel("Average Arrival Delay")
plt.tight_layout()
plt.savefig("figures/time_of_day_delay.png", dpi=300, bbox_inches="tight")
plt.close()

carrier_delay = df.groupby("OP_CARRIER")["ARR_DELAY"].mean().sort_values()
plt.figure(figsize=(12, 6))
sb.barplot(x=carrier_delay.index, y=carrier_delay.values)
plt.xticks(rotation=90)
plt.title("Average Arrival Delay by Carrier")
plt.xlabel("Carrier")
plt.ylabel("Average Arrival Delay")
plt.tight_layout()
plt.savefig("figures/carrier_delay.png", dpi=300, bbox_inches="tight")
plt.close()

carrier_cancel = df.groupby("OP_CARRIER")["CANCELLED"].mean().mul(100).sort_values()
plt.figure(figsize=(12, 5))
sb.barplot(x=carrier_cancel.index, y=carrier_cancel.values)
plt.title("Cancellation Rate by Carrier")
plt.xlabel("Carrier")
plt.ylabel("Cancellation Rate (%)")
plt.tight_layout()
plt.savefig("figures/carrier_cancel_rate.png", dpi=300, bbox_inches="tight")
plt.close()

airport_analysis = df.groupby("ORIGIN").agg({"ARR_DELAY": "mean", "FL_DATE": "count"}).rename(columns={"FL_DATE": "FLIGHT_COUNT"})
top_airports = airport_analysis.sort_values("FLIGHT_COUNT", ascending=False).head(10)
plt.figure(figsize=(12, 5))
sb.barplot(data=top_airports.sort_values("ARR_DELAY"), x="ARR_DELAY", y=top_airports.sort_values("ARR_DELAY").index)
plt.title("Average Arrival Delay at Busiest Origin Airports")
plt.xlabel("Average Arrival Delay")
plt.ylabel("Airport")
plt.tight_layout()
plt.savefig("figures/airport_delays.png", dpi=300, bbox_inches="tight")
plt.close()

sample_df = df.sample(200000, random_state=42)
plt.figure(figsize=(10, 6))
plt.hexbin(sample_df["DEP_DELAY"], sample_df["ARR_DELAY"], gridsize=60, cmap="viridis", mincnt=1)
plt.colorbar(label="Flight Count")
plt.xlabel("Departure Delay")
plt.ylabel("Arrival Delay")
plt.title("Departure Delay vs Arrival Delay")
plt.tight_layout()
plt.savefig("figures/dep_arr_hexbin.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
sb.scatterplot(data=sample_df, x="DISTANCE", y="ARR_DELAY", alpha=0.3)
plt.title("Distance vs Arrival Delay")
plt.xlabel("Distance")
plt.ylabel("Arrival Delay")
plt.tight_layout()
plt.savefig("figures/distance_delay.png", dpi=300, bbox_inches="tight")
plt.close()

delay_cols = ["DEP_DELAY", "ARR_DELAY", "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
corr_matrix = df[delay_cols].corr()
plt.figure(figsize=(10, 6))
sb.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Delay Components")
plt.tight_layout()
plt.savefig("figures/corr_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

delay_columns = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
delay_data = df[delay_columns].fillna(0)
df["MAIN_DELAY_CAUSE"] = delay_data.idxmax(axis=1)
df.loc[delay_data.sum(axis=1) == 0, "MAIN_DELAY_CAUSE"] = "NO_REPORTED_DELAY"

delayed_flights = df[df["ARR_DELAY"] >= 15]
main_cause_counts = delayed_flights["MAIN_DELAY_CAUSE"].value_counts()
plt.figure(figsize=(10, 5))
sb.barplot(x=main_cause_counts.index, y=main_cause_counts.values)
plt.title("Dominant Delay Causes")
plt.xlabel("Delay Cause")
plt.ylabel("Number of Flights")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("figures/main_delay_cause.png", dpi=300, bbox_inches="tight")
plt.close()

print("All figures generated and saved to figures/")
