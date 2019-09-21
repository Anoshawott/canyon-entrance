# Understanding Fraud Detection

## By Anosh.S

Using python this python notebook will automatically identify fraudulent credit card transactions utilising processes in machine learning.

As economic crime grows the concern for fraudulent activity has risen. As a result, the recent developments in deep learning, artificial intelligence and machine learning methods has facilitated many means for automating such a process. Through a systematic identification of outliers, developing this pythonic approach, potential anomalies can be understood and dealt with accordingly.

```
# Target variable visual comparison
fc = train_trn['isFraud'].value_counts(normalize = True).to_frame()
fc.plot.bar()
fc.T
```

![Annotation 2019-09-21 181300](https://user-images.githubusercontent.com/54537931/65370388-80c5d080-dc9b-11e9-80fa-4913695714e8.jpg)

```
# Correlation Heatmap of features C1 to C14

ccols = ['C%d' % number for number in range(1,15)]

plt.figure(figsize = (10,5))

corr = train_trn[['isFraud'] + ccols].corr()
sns.heatmap(corr, annot = True, fmt = '.2f')
```

![Annotation 2019-09-21 181416](https://user-images.githubusercontent.com/54537931/65370397-a94dca80-dc9b-11e9-9603-b1ccb2c3e934.jpg)
