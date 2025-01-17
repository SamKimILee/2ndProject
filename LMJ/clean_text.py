import re

from hanspell import spell_checker

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', '\u200b': '', '…': '...', '\ufeff': '', 'करना': '', 'है': ''} 

'''
=================================================================================
spell_key : 네이버 맞춤법 검사기에서 검사하기 누른 다음 Network 탭에서
            "SpellerProxy"로 시작하는 패킷의 Payload 탭에서 Parameter값 가지고 오기
spell_key = '[passPortKey]', '[_]'
=================================================================================
'''

def clean_str(text, spell_key):

    # 특수 문자, 줄바꿈, 불필요한 공백 등 제거 및 치환
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f'{p} ')

    text = re.sub('([ㄱ-ㅎㅏ-ㅣ]+)', repl='', string=text)  # 한글 자음, 모음 제거
    text = re.sub(r'\b\w*\d\w*\b', repl='', string=text) # 숫자와 숫자 포함된 단어 제거
    # text = re.sub('([0-9]+)', repl='', string=text) # 주의! '22살인 ~' -> '살인 ~'
    text = re.sub('[^\w\s\n]', repl='', string=text)  # 특수기호제거
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', string=text)
    text = re.sub('\n', '.', string=text)
    text = re.sub(r'\s+', ' ', text).strip()

    # hanspell라이브러리로 맞춤법 검사 - 300자까지 가능 (초과 시 False)
    spell_checker.setParam(spell_key[0], spell_key[1])
    text = spell_checker.check(text.strip())

    return text.checked

'''
------------------------------------------------------------------
check 반환값: Checked(result=True, original='하기 실타', checked='하기 싫다',
             errors=1, words=OrderedDict([('하기', 0), ('싫다', 0)]), time=0.039252281188964844)
------------------------------------------------------------------
맞춤법 검사 성고 여부 (spelled_text.result): True
검사 전 문장 (spelled_text.original): 하기 실타
검사 후 문장 (spelled_text.checked): 하기 싫다
맞춤법 오류 수 (spelled_text.errors): 1
공백으로 구분한 단어와 오류 OrderedDict (spelled_text.words): OrderedDict([('하기', 0), ('싫다', 0)])
총 요청 시간 (spelled_text.time): 0.039252281188964844
------------------------------------------------------------------
words의 오류 코드 값
0 : PASSED
1 : WRONG_SPELLING (맞춤법 오류)
2 : WRONG_SPACING (띄어쓰기 오류)
3 : AMBIGUOUS (표준어가 의심되는 단어 또는 구절)
4 : STATISTICAL_CORRECTION (통계적 교정에 따른 단오 또는 구절)
------------------------------------------------------------------
'''