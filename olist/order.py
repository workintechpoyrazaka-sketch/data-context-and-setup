import pandas as pd
import numpy as np
from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''
    def __init__(self):
        # Assign an attribute ".data" to all new instances of Order
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True):
        """
        Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status]
        and filters out non-delivered orders unless specified
        """
        orders = self.data['orders'].copy()

        if is_delivered:
            orders = orders.query("order_status == 'delivered'").copy()

        orders['order_purchase_timestamp'] = \
            pd.to_datetime(orders['order_purchase_timestamp'])
        orders['order_delivered_customer_date'] = \
            pd.to_datetime(orders['order_delivered_customer_date'])
        orders['order_estimated_delivery_date'] = \
            pd.to_datetime(orders['order_estimated_delivery_date'])

        orders['wait_time'] = \
            (orders['order_delivered_customer_date']
             - orders['order_purchase_timestamp']) / np.timedelta64(1, 'D')

        orders['expected_wait_time'] = \
            (orders['order_estimated_delivery_date']
             - orders['order_purchase_timestamp']) / np.timedelta64(1, 'D')

        def compute_delay(row):
            late = (row['order_delivered_customer_date']
                    - row['order_estimated_delivery_date']) / np.timedelta64(1, 'D')
            if late > 0:
                return late
            return 0.0

        orders['delay_vs_expected'] = orders.apply(compute_delay, axis=1)

        return orders[['order_id', 'wait_time', 'expected_wait_time',
                       'delay_vs_expected', 'order_status']]

    def get_review_score(self):
        """
        Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """
        reviews = self.data['order_reviews'].copy()

        reviews['dim_is_five_star'] = \
            reviews['review_score'].map(lambda score: 1 if score == 5 else 0)
        reviews['dim_is_one_star'] = \
            reviews['review_score'].map(lambda score: 1 if score == 1 else 0)

        return reviews[['order_id', 'dim_is_five_star',
                        'dim_is_one_star', 'review_score']]

    def get_number_items(self):
        """
        Returns a DataFrame with:
        order_id, number_of_items
        """
        items = self.data['order_items'].copy()

        return items.groupby('order_id', as_index=False)['order_item_id'] \
            .count() \
            .rename(columns={'order_item_id': 'number_of_items'})

    def get_number_sellers(self):
        """
        Returns a DataFrame with:
        order_id, number_of_sellers
        """
        items = self.data['order_items'].copy()

        return items.groupby('order_id', as_index=False)['seller_id'] \
            .nunique() \
            .rename(columns={'seller_id': 'number_of_sellers'})

    def get_price_and_freight(self):
        """
        Returns a DataFrame with:
        order_id, price, freight_value
        """
        items = self.data['order_items'].copy()

        return items.groupby('order_id', as_index=False)[['price', 'freight_value']] \
            .sum()

    # Optional
    def get_distance_seller_customer(self):
        """
        Returns a DataFrame with:
        order_id, distance_seller_customer
        """
        data = self.data
        orders = data['orders']
        order_items = data['order_items']
        sellers = data['sellers']
        customers = data['customers']

        geo = data['geolocation']
        geo = geo.groupby('geolocation_zip_code_prefix', as_index=False).first()

        sellers_geo = sellers.merge(
            geo,
            how='left',
            left_on='seller_zip_code_prefix',
            right_on='geolocation_zip_code_prefix'
        )[['seller_id', 'seller_zip_code_prefix',
           'geolocation_lat', 'geolocation_lng']] \
            .rename(columns={'geolocation_lat': 'seller_lat',
                             'geolocation_lng': 'seller_lng'})

        customers_geo = customers.merge(
            geo,
            how='left',
            left_on='customer_zip_code_prefix',
            right_on='geolocation_zip_code_prefix'
        )[['customer_id', 'customer_zip_code_prefix',
           'geolocation_lat', 'geolocation_lng']] \
            .rename(columns={'geolocation_lat': 'customer_lat',
                             'geolocation_lng': 'customer_lng'})

        matching_geo = order_items \
            .merge(sellers_geo, on='seller_id') \
            .merge(orders, on='order_id') \
            .merge(customers_geo, on='customer_id') \
            .dropna()

        matching_geo['distance_seller_customer'] = matching_geo.apply(
            lambda row: haversine_distance(
                row['seller_lng'], row['seller_lat'],
                row['customer_lng'], row['customer_lat']),
            axis=1)

        return matching_geo.groupby('order_id', as_index=False) \
            .agg({'distance_seller_customer': 'mean'})

    def get_training_data(self,
                          is_delivered=True,
                          with_distance_seller_customer=False):
        """
        Returns a clean DataFrame (without NaN), with the all following columns:
        ['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected',
        'order_status', 'dim_is_five_star', 'dim_is_one_star', 'review_score',
        'number_of_items', 'number_of_sellers', 'price', 'freight_value',
        'distance_seller_customer']
        """
        training_set = self.get_wait_time(is_delivered) \
            .merge(self.get_review_score(), on='order_id') \
            .merge(self.get_number_items(), on='order_id') \
            .merge(self.get_number_sellers(), on='order_id') \
            .merge(self.get_price_and_freight(), on='order_id')

        if with_distance_seller_customer:
            training_set = training_set.merge(
                self.get_distance_seller_customer(), on='order_id')

        return training_set.dropna()
