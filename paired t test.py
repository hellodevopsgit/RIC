from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("../RICFiles/blood_pressure.csv")
print(df[['bp_before','bp_after']].describe())
df[['bp_before', 'bp_after']].plot(kind='box')
plt.show()
plt.savefig('paired_t-test/boxplot_outliers.png'
df['bp_difference'] = df['bp_before'] - df['bp_after']
df['bp_difference'].plot(kind='hist', title= 'Blood Pressure Difference Histogram')
plt.show()
plt.savefig('paired_t-test/blood pressure difference histogram.png')
stats.probplot(df['bp_difference'], plot= plt)
plt.title('Blood pressure Difference Q-Q Plot')
plt.savefig('paired_t-test/blood pressure difference qq plot.png')
plt.show()
stats.shapiro(df['bp_difference'])
stats.ttest_rel(df['bp_before'], df['bp_after'])
