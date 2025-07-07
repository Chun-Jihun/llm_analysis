import requests
import time
import pandas as pd
from datetime import datetime
from collections import Counter # 중복 ID 분석을 위해 유지 (검증용)

def get_all_steam_reviews(app_id=3430470, language='korean', filter_type='all', review_limit=None):
    all_reviews_data = []
    seen_recommendation_ids = set() # 이미 수집된 리뷰의 recommendationid를 저장하는 집합
    cursor = '*'
    page_num = 0

    while True:
        page_num += 1
        
        if review_limit is not None and len(all_reviews_data) >= review_limit:
            print(f"리뷰 수집 한도({review_limit}개)에 도달하여 루프를 시작하지 않습니다.")
            break

        api_url = f"https://store.steampowered.com/appreviews/{app_id}"
        
        num_to_fetch_this_call = 100
        if review_limit is not None:
            remaining_needed = review_limit - len(all_reviews_data)
            if remaining_needed <= 0:
                break 
            num_to_fetch_this_call = min(num_to_fetch_this_call, remaining_needed)

        params = {
            'json': '1',
            'filter': filter_type,
            'language': language,
            'day_range': '0',
            'cursor': cursor.encode('utf-8'),
            'review_type': 'all',
            'purchase_type': 'all',
            'num_per_page': str(num_to_fetch_this_call)
        }

        try:
            goal_text = f"(목표: {review_limit}개)" if review_limit is not None else "(목표: 모든 리뷰)"
            print(f"페이지 {page_num} 리뷰 요청... (현재 {len(all_reviews_data)}개 수집, 이번 요청 최대 {num_to_fetch_this_call}개) {goal_text}")
            
            response = requests.get(api_url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            if data.get('success') == 1:
                if page_num == 1: # 첫 페이지 응답일 때만 query_summary 출력 (디버깅용)
                    query_summary_first_page = data.get('query_summary', {})
                    print("--- API Query Summary (첫 페이지) ---")
                    print(f"  요청된 언어 '{language}'에 대한 정보:")
                    print(f"  - 이번 응답 리뷰 수 (API 보고): {query_summary_first_page.get('num_reviews')}")
                    print(f"  - 전체 리뷰 수 (해당 언어, API 보고): {query_summary_first_page.get('total_reviews')}")
                    print("------------------------------------")

                current_reviews_from_api = data.get('reviews', [])
                query_summary = data.get('query_summary', {})
                num_reviews_on_page_api = int(query_summary.get('num_reviews', 0))

                if not current_reviews_from_api: # API가 빈 리스트를 반환한 경우
                    if num_reviews_on_page_api == 0: # query_summary에서도 0개라고 명시
                        print(f"API가 페이지 {page_num}에 리뷰 0개를 반환하여 수집을 중단합니다.")
                    else: # 드문 경우지만, 리스트는 비었는데 num_reviews는 0이 아닌 경우
                        print(f"페이지 {page_num}에 리뷰 데이터가 없으나, API는 {num_reviews_on_page_api}개를 보고했습니다. 수집 중단.")
                    break
                
                # 현재 페이지에서 새롭고 고유한 리뷰만 필터링
                new_unique_reviews_this_page = []
                page_had_only_duplicates = True # 현재 페이지가 모두 중복이라고 가정
                
                for review in current_reviews_from_api:
                    rec_id = review.get('recommendationid')
                    # recommendationid가 없는 리뷰는 일단 고유하다고 간주 (매우 드문 케이스)
                    if rec_id is None or rec_id not in seen_recommendation_ids:
                        new_unique_reviews_this_page.append(review)
                        if rec_id is not None:
                            seen_recommendation_ids.add(rec_id)
                        page_had_only_duplicates = False # 새로운 리뷰가 하나라도 있으면 False
                
                if page_had_only_duplicates and current_reviews_from_api: # 페이지에 리뷰가 있었지만, 모두 이전에 본 것들인 경우
                    print(f"페이지 {page_num}의 모든 리뷰({len(current_reviews_from_api)}개)가 이전에 수집된 리뷰와 중복되어 수집을 중단합니다.")
                    break

                # 실제 추가된 새 리뷰가 없다면 (예: 페이지가 비었거나, 모두 중복이었음) 더 진행할 필요 없음
                if not new_unique_reviews_this_page:
                    if current_reviews_from_api: # 페이지에 리뷰는 있었으나, 유니크한 새 리뷰가 없었음 (위의 all_duplicates_on_page에서 이미 처리됨)
                        pass # 이미 위에서 처리됨
                    else: # 페이지 자체가 비어있었음 (이것도 위에서 처리됨)
                        print(f"페이지 {page_num}에서 새로운 고유 리뷰를 찾지 못했습니다.")
                    # 이 경우 다음 커서가 있어도 의미가 없을 수 있으나, 일단은 API의 다음 커서 유무로 판단
                else: # 새로운 고유 리뷰가 있는 경우에만 추가
                    print(f"페이지 {page_num}에서 {len(new_unique_reviews_this_page)}개의 새로운 고유 리뷰 발견.")
                    if review_limit is not None:
                        space_left = review_limit - len(all_reviews_data)
                        if space_left <= 0:
                             print(f"리뷰 한도({review_limit}개)에 이미 도달하여 더 이상 추가하지 않습니다.")
                             break # 루프 종료
                        reviews_to_add_count = min(len(new_unique_reviews_this_page), space_left)
                        all_reviews_data.extend(new_unique_reviews_this_page[:reviews_to_add_count])
                        if reviews_to_add_count < len(new_unique_reviews_this_page):
                             print(f"리뷰 한도 도달로 인해 현재 페이지의 새 리뷰 중 일부만 추가됨 ({reviews_to_add_count}/{len(new_unique_reviews_this_page)}).")
                    else: # review_limit이 없을 경우
                        all_reviews_data.extend(new_unique_reviews_this_page)
                
                cursor = data.get('cursor')

                # 최종 중단 조건 확인
                if review_limit is not None and len(all_reviews_data) >= review_limit:
                    print(f"목표 리뷰 수 ({review_limit}개)에 도달했거나 초과하여 수집을 중단합니다.")
                    break
                if not cursor:
                    print("API 응답에 다음 커서가 없어 수집을 중단합니다 (마지막 페이지).")
                    break
                
                time.sleep(1.5)

            else: # API success != 1
                print(f"API 요청 실패 (success != 1): {data.get('success')}")
                print(f"요약: {data.get('query_summary')}")
                if data.get('success') == 2 and not data.get('reviews'): # 보통 결과가 없을 때
                     print("API에서 더 이상 리뷰가 없다고 응답했습니다 (success: 2).")
                break 
        
        except requests.exceptions.Timeout:
            print(f"HTTP 요청 시간 초과 (페이지 {page_num}). 현재까지 {len(all_reviews_data)}개 수집.")
            break 
        except requests.exceptions.RequestException as e:
            print(f"HTTP 요청 중 오류 발생 (페이지 {page_num}): {e}")
            break 
        except ValueError: 
            print(f"응답 분석 중 오류 발생 (JSON 형식 아님, 페이지 {page_num}). 응답 내용: {response.text}")
            break
            
    # 최종적으로 review_limit을 초과한 경우 잘라내기 (안전장치)
    if review_limit is not None and len(all_reviews_data) > review_limit:
        all_reviews_data = all_reviews_data[:review_limit]
        
    return all_reviews_data


if __name__ == "__main__":
    app_id_input = input("리뷰를 가져올 Steam 앱 ID를 입력하세요: ")

    if app_id_input and app_id_input.isdigit():
        target_app_id = int(app_id_input)
        
        selected_language = input("리뷰 언어를 입력하세요 (예: korean, english, all - 기본값 korean): ")
        if not selected_language:
            selected_language = 'korean'
        
        # 가져올 리뷰 수 설정 (예시: 600개 또는 None으로 설정하여 모두 가져오기)
        MAX_REVIEWS_TO_FETCH = 600
        # MAX_REVIEWS_TO_FETCH = None # 모든 리뷰를 가져오도록 하려면 None으로 설정
        limit_text = f"최대 {MAX_REVIEWS_TO_FETCH}개" if MAX_REVIEWS_TO_FETCH is not None else "모든"

        print(f"\n앱 ID {target_app_id}의 '{selected_language}' 언어 {limit_text} 리뷰를 가져옵니다...")
        
        all_fetched_reviews = get_all_steam_reviews(
            target_app_id,
            language=selected_language,
            review_limit=MAX_REVIEWS_TO_FETCH
        )

        if all_fetched_reviews:
            print(f"\n총 {len(all_fetched_reviews)}개의 고유한 리뷰를 수집했습니다.")

            # --- 최종 중복 확인 코드 (검증용) ---
            # 이 시점에서는 all_fetched_reviews 리스트 내에는 recommendationid 기준 중복이 없어야 합니다.
            recommendation_ids = [review.get('recommendationid') for review in all_fetched_reviews if review.get('recommendationid') is not None]
            if not recommendation_ids and all_fetched_reviews: # 리뷰는 있는데 ID가 없는 경우 (매우 드묾)
                 print("수집된 리뷰에 recommendationid가 없어 최종 중복 검사를 정확히 수행할 수 없습니다.")
            elif recommendation_ids: # ID가 있는 리뷰가 있을 때만 검사
                unique_ids_count = len(set(recommendation_ids))
                print(f"최종 수집된 리뷰의 recommendationid 기준 분석 (검증용):")
                print(f"  - 전체 recommendationid 수: {len(recommendation_ids)}")
                print(f"  - 고유 recommendationid 수: {unique_ids_count}")

                if len(recommendation_ids) != unique_ids_count:
                    # 이 메시지가 나타나면 get_all_steam_reviews 내부의 중복 제거 로직에 문제가 있는 것입니다.
                    print(f"  [오류!] 최종 결과에 중복된 recommendationid가 {len(recommendation_ids) - unique_ids_count}개 포함되어 있습니다. 로직 확인 필요.")
                    id_counts = Counter(recommendation_ids)
                    duplicated_ids_details = {id_val: count for id_val, count in id_counts.items() if count > 1}
                    print(f"  - 중복된 ID 및 횟수: {duplicated_ids_details}")
                else:
                    print("  - 최종 결과의 모든 recommendationid가 고유합니다. (예상대로 작동)")
            # --- 중복 확인 코드 끝 ---

            processed_reviews = []
            for review_data in all_fetched_reviews:
                author_info = review_data.get('author', {})
                processed_review = {
                    'recommendation_id': review_data.get('recommendationid'),
                    'steam_id': author_info.get('steamid'),
                    'num_games_owned': author_info.get('num_games_owned'),
                    'num_reviews_by_author': author_info.get('num_reviews'),
                    'playtime_forever_minutes': author_info.get('playtime_forever'),
                    'playtime_last_two_weeks_minutes': author_info.get('playtime_last_two_weeks'),
                    'last_played_timestamp': author_info.get('last_played'),
                    'language': review_data.get('language'),
                    'review_text': review_data.get('review'),
                    'timestamp_created': review_data.get('timestamp_created'),
                    'timestamp_updated': review_data.get('timestamp_updated'),
                    'voted_up': review_data.get('voted_up'),
                    'votes_up': review_data.get('votes_up'),
                    'votes_funny': review_data.get('votes_funny'),
                    'weighted_vote_score': review_data.get('weighted_vote_score'),
                    'comment_count': review_data.get('comment_count'),
                    'steam_purchase': review_data.get('steam_purchase'),
                    'received_for_free': review_data.get('received_for_free'),
                    'written_during_early_access': review_data.get('written_during_early_access')
                }
                processed_reviews.append(processed_review)
            
            if not processed_reviews:
                print("처리할 리뷰 데이터가 없습니다. CSV 파일을 생성하지 않습니다.")
            else:
                df = pd.DataFrame(processed_reviews)

                for col in ['timestamp_created', 'timestamp_updated', 'last_played_timestamp']:
                    if col in df.columns:
                        df[col + '_readable'] = pd.to_datetime(df[col], unit='s', errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                
                if 'playtime_forever_minutes' in df.columns:
                    df['playtime_forever_hours'] = (df['playtime_forever_minutes'] / 60).round(1)
                if 'playtime_last_two_weeks_minutes' in df.columns:
                    df['playtime_last_two_weeks_hours'] = (df['playtime_last_two_weeks_minutes'] / 60).round(1)

                limit_suffix = f"_limit{MAX_REVIEWS_TO_FETCH}" if MAX_REVIEWS_TO_FETCH is not None else "_all"
                import datetime
                today = datetime.date.today()
                csv_filename = f'steam_reviews_{today.strftime("%y%m%d")}.csv' # 파일명에 unique 추가
                try:
                    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
                    print(f"\n고유한 리뷰 데이터를 '{csv_filename}' 파일로 성공적으로 저장했습니다.")
                except Exception as e:
                    print(f"\nCSV 파일 저장 중 오류 발생: {e}")
        else:
            print("리뷰를 가져오지 못했거나 해당 조건의 리뷰가 없습니다.")
    else:
        print("유효한 숫자 형식의 앱 ID를 입력해주세요.")