from stats_extractor import StatsExtractor


def main():
    train, test = pd.read_csv(DATA_RAW / 'train.csv'), pd.read_csv(DATA_RAW / 'test.csv')
    train, test = StatsExtractor().fit_transform(train, test)
    test.reset_index(drop=True, inplace=True)
    train.to_feather(DATA_PROCESSED / 'train_processed.feather')
    test.to_feather(DATA_PROCESSED / 'test_processed.feather')


if __name__ == '__main__':
    main()
