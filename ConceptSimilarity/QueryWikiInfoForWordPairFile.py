#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
2017-5-25
对这种显示为AI，而实质为AI (disambiguation)的情况进行修正
ReTag_wikinotedisambigpage_2 = re.compile(r'<a href=".*? class=".*?mw-disambig.*?".*? title="(.*?)".*?>.*?</a>',re.IGNORECASE)#<a href="/wiki/AI_(disambiguation)" class="mw-redirect mw-disambig" title="AI (disambiguation)">AI</a>  

2017-5-24b


2017-5-24a
原版本对消歧页的侯选概念的处理，直接去使用的disambigerror.options提供的列表发现其并不准确。
比如frame这一个字，前方的structural system和steel frame都没有返回；此外，返回的列表包括See also感觉其中的概念关系不大，应该删除。
故在原基础上，对有关消歧页的选项的处理做了修改，添加了collectItemOnDisambigWikiPage()函数，返回自己分析得到的候选列表 及 see also中的无关概念列表。

2017-5-15
在原基础上，做两个工作：
1、文本的输出都改为utf-8格式
2、将链表的输出改一下，都改为\t分隔
'''

import wikipedia

import fileinput
import re
import argparse
import os
import sys
import time
import copy

import requests
from bs4 import BeautifulSoup
from bs4 import NavigableString


ReTag_wikinote = re.compile(r'<div role="note" class="hatnote(?:.*?)">(.*?)</div>',re.IGNORECASE) #<div role="note" class="hatnote">For the hypotheses of a theorem, see <a href="/wiki/Theorem" title="Theorem">Theorem</a>. For other uses, see <a href="/wiki/Hypothesis_(disambiguation)" class="mw-disambig" title="Hypothesis (disambiguation)">Hypothesis (disambiguation)</a>.</div>
ReTag_wikinotelinks = re.compile(r'(<a href=(?:.*?)</a>)',re.IGNORECASE)#For the hypotheses of a theorem, see <a href="/wiki/Theorem" title="Theorem">Theorem</a>. For other uses, see <a href="/wiki/Hypothesis_(disambiguation)" class="mw-disambig" title="Hypothesis (disambiguation)">Hypothesis (disambiguation)</a>.
ReTag_wikinotedisambigpage = re.compile(r'<a href=".*? class=".*?mw-disambig.*?" .*?>(.*?)</a>',re.IGNORECASE)#<a href="/wiki/Hypothesis_(disambiguation)" class="mw-disambig" title="Hypothesis (disambiguation)">Hypothesis (disambiguation)</a> /
ReTag_wikinotedisambigpage_2 = re.compile(r'<a href=".*? class=".*?mw-disambig.*?".*? title="(.*?)".*?>.*?</a>',re.IGNORECASE)#<a href="/wiki/AI_(disambiguation)" class="mw-redirect mw-disambig" title="AI (disambiguation)">AI</a>
#<a href="/wiki/Rocket_ship_(disambiguation)" class="mw-redirect mw-disambig" title="Rocket ship (disambiguation)">Rocket ship (disambiguation)</a>
ReTag_notconfusedwith = re.compile(r'(?:Not to be confused with <a)', re.IGNORECASE) #Ecuador:  Not to be confused with <a href="/wiki/Equator" title="Equator">Equator</a>.

ReTag_wikinoteotherpage = re.compile(r'<a href=".*?">(.*?)</a>',re.IGNORECASE)#<a href="/wiki/Theorem" title="Theorem">Theorem</a>
ReTag_disambigchar = re.compile(r'\(disambiguation\)$',re.IGNORECASE) # Starship (disambiguation)  https://en.wikipedia.org/wiki/Spaceship_(disambiguation)
ReTag_listofchar = re.compile(r'(List of)',re.IGNORECASE) #List of Latin phrases (M)

def lines_from(input):
    for line in input:
        line = line.decode('utf-8')
        line = line.strip('\n\r')
        yield line


def get_cur_info():
    """Return the frame object for the caller's stack frame."""
    try:
        raise Exception
    except:
        f = sys.exc_info()[2].tb_frame.f_back
    return (f.f_code.co_name, f.f_lineno)


def listitems2strs(sourcelist):
    if sourcelist == None:
        return u'None'
    ret = []
    for item in sourcelist:
        ret.append(item)
    ret = '\t'.join(ret)
    return ret



def FormatOutputofWikiInfo(a):
    '''
    :param a: the dictionary outputted by WikiInfo function
    :return: return its string
    '''
    assert len(a.items()) == 12, u'{}: the following code need to update to satisfy with the data of a'.format(get_cur_info())

    b = copy.deepcopy(a)

    b[u'Title'] = b[u'Title']
    b[u'bNormal'] = b[u'bNormal']
    b[u'bDisambig'] = b[u'bDisambig']
    b[u'bRedirect'] = b[u'bRedirect']
    b[u'RedirectTitle'] = b[u'RedirectTitle']
    b[u'bPageErr'] = b[u'bPageErr']
    b[u'bHttpTimeOut'] = b[u'bHttpTimeOut']
    b[u'DisambigItems'] = listitems2strs(b[u'DisambigItems'])
    b[u'HatnoteLinkItems'] = listitems2strs(b[u'HatnoteLinkItems'])
    b[u'HatnoteLinkDisambig'] = listitems2strs(b[u'HatnoteLinkDisambig'])
    b[u'HatnoteLinkDisambigItems'] = listitems2strs(b[u'HatnoteLinkDisambigItems'])

    #ret = u'Title:{}    bNormal:{}    bDisambig:{}    bRedirect:{}    RedirectTitle:{}    bPageErr:{}    bHttpTimeOut:{}    DisambigItems:{}    HatnoteLinkItems:{}    HatnoteLinkDisambig:{}    HatnoteLinkDisambigItems:{}'.format\
    #(a[u'Title'], a[u'bNormal'], a[u'bDisambig'], a[u'bRedirect'], a[u'RedirectTitle'], a[u'bPageErr'], a[u'bHttpTimeOut'], a[u'DisambigItems'], a[u'HatnoteLinkItems'], a[u'HatnoteLinkDisambig'], a[u'HatnoteLinkDisambigItems'])
    ret = u'Title:{}    bNormal:{}    bDisambig:{}    bRedirect:{}    RedirectTitle:{}    bPageErr:{}    bHttpTimeOut:{}    DisambigItems:{}    HatnoteLinkItems:{}    HatnoteLinkDisambig:{}    HatnoteLinkDisambigItems:{}    Exception:{}'.format \
        (b[u'Title'], b[u'bNormal'], b[u'bDisambig'], b[u'bRedirect'], b[u'RedirectTitle'], b[u'bPageErr'],
         b[u'bHttpTimeOut'], b[u'DisambigItems'], b[u'HatnoteLinkItems'], b[u'HatnoteLinkDisambig'],
         b[u'HatnoteLinkDisambigItems'], b[u'Exception'])


def GetWikiAmbiguousTitles(ambiguousword,redirect=True):
    '''
    :param ambiguousword: it should be a ambigous word, such as:Hypothesis (disambiguation). otherwise, reutrn itself
    :return: the refered wiki titles by the ambiguousword
    '''
    ret = []
    try:
        wikiret = wikipedia.page(ambiguousword,auto_suggest=False,redirect=redirect)
        ret.append(wikiret.title)
        return ret
    except wikipedia.exceptions.DisambiguationError as disambigerror:
        ret = disambigerror.options
        return ret
    except Exception as inst:
        print(type(inst))
        print inst.args
        print inst.message
        print inst
        print "ambigousword: " + ambiguousword + " , induces a Exception! (GetWikiAmbiguousTitles(ambiguousword,redirect=True))"
        return ret
    return ret


def WikiInfo(text, retrytimes=5, permitRedirectQuery=True, lookforfinalpage=True):
    '''

    :param text: the title that need to look for in wikipedia
    :param retrytimes: the permitted times to retry (avoid too many httptimeout error)
    :param permitRedirectQuery: 如果text对应一个重定向页，是否把重定向页的Disambig, HatnoteLinkDisambig, HatnoteLinkItems等一系列信息返回。y,返回；n,不返回
    :param lookforfinalpage:对于每个返回的wiki的title，是否要找到其终极页面（有些页面，可能需要多次重定向）
    :return:dictret['Title']  返回当前的查询词text
            dictret['bNormal'] 查询词是否对应一个正常、唯一确定的页面
            dictret['bRedirect'] 查询词是否对应一个重定向页面
            dictret['RedirectTitle'] 如果对应一个重定向页面，其终极(finalpage)的title是什么
            dictret['bPageErr'] 查询词是否在wikipedia中不存在
            dictret['bHttpTimeOut'] 是否存在网络访问故障
            dictret['bDisambig'] A是否对应一个消歧页。如果bRedirect为True，A指当前查询词重定向的终极title；如果bRedirect为False，A指当前查询词。
            dictret['DisambigItems'] 如果A对应一个消歧页，该消歧页包含哪些可能的概念条目
            dictret['HatnoteLinkDisambig'] 如果A页面的HatNote部分，存在消歧页链接，则把它们记录在这里。如果bRedirect为False，则A指查询词对应的页面；如果bRedirect为True，则A指终级重定向页面。如Spacecraft,hypertheis
            dictret['HatnoteLinkDisambigItems'] 对应dictret['HatnoteLinkDisambig']的消歧页中包含的可能的概念条目。
            dictret['HatnoteLinkItems'] 如果A页面的HatNote部分，存在其他一些可能的链接，则把它们记录在这里。如果bRedirect为False，则A指查询词对应的页面；如果bRedirect为True，则A指终级重定向页面。

如果['bNormal'] =True，意味着当前查询词对应 一个正常页面
如果['bDisambig']=True and ['bRedirect']=False，意味着当前查询词对应 一个消歧页
如果['bRedirect']=True，意味着['bDisambig'] ['DisambigItems'] ['bPageErr'] ['bHttpTimeOut'] ['HatnoteLinkDisambig'] ['HatnoteLinkDisambigItems'] ['HatnoteLinkItems'] 这几个属性均不再是对应当前查询词的，而是对应查询词的终极重定向概念的。
    '''

    dictret = {u'Title': text}

    repeattimes = 0

    while True:
        # Title,  DisambigItems, RedirectText   are the candidate related concept at first place
        # HatnoteLinkDisambigItems, HatnoteLinkItems   are the candidate related concept at second place
        dictret[u'bNormal'] = None # is this a normal wiki page
        dictret[u'bDisambig'] = None # is this a disambiguation page
        dictret[u'DisambigItems'] = None # the refered titles with current ambiguous title
        dictret[u'bRedirect'] = None # is there a redirect action for current title
        dictret[u'RedirectTitle'] = None # the redirect title
        dictret[u'bPageErr'] = None # is there a page error
        dictret[u'bHttpTimeOut'] = None # is there a Internet network problem
        dictret[u'HatnoteLinkDisambig'] = None # if there is a ambigous link in HatNote of the wikipage, such as 'hypothesis'
        dictret[u'HatnoteLinkDisambigItems'] = None # if there is a ambigous link in HatNote of the wikipage, such as 'hypothesis', get its detailed referred items.
        dictret[u'HatnoteLinkItems'] = None # if there is some non-ambigous links in HatNote of the wikipage, such as 'hypothesis'
        dictret[u'Exception'] = None  # 记录未知错误


        repeattimes += 1
        if repeattimes > retrytimes:
            print 'wikipedia can\'t be accessed normally! please check Interent'
            dictret[u'Exception'] = 'wikipedia can\'t be accessed normally! please check Interent'
            break

        try:
            dictret[u'bNormal'] = True
            wikipedia.set_lang('en')
            wikiret = wikipedia.page(text, auto_suggest=False, redirect=False)

            #get more HatNote information about normal wiki page
            Note = wikiret.html()[0:1500]
            hatnotelinks=[]
            hatnotelinkdisambig=[]
            hatnotelinkdisambigitems=[]
            for m in ReTag_wikinote.finditer(Note):
                t = m.group(1)#For the hypotheses of a theorem, see <a href="/wiki/Theorem" title="Theorem">Theorem</a>. For other uses, see <a href="/wiki/Hypothesis_(disambiguation)" class="mw-disambig" title="Hypothesis (disambiguation)">Hypothesis (disambiguation)</a>.
                if ReTag_notconfusedwith.search(t): #说明这是一个 提醒不要误解的页面。如果Ecuador的页面中的Not to be confused with Equator.
                    continue
                for linkmatch in ReTag_wikinotelinks.finditer(t):
                    t2 = linkmatch.group(1)  #<a href="/wiki/Theorem" title="Theorem">Theorem</a>     #<a href="/wiki/Hypothesis_(disambiguation)" class="mw-disambig" title="Hypothesis (disambiguation)">Hypothesis (disambiguation)</a>
                    bFlagAmbiguous = False

                    disambigmatch = ReTag_wikinotedisambigpage.search(t2)
                    #-----
                    disambigmatch_2 = ReTag_wikinotedisambigpage_2.search(t2)
                    if disambigmatch:#从我设计的本意上说，当disambigmatch不为空时，disambigmatch_2按说也不能为空。如果违反，则报错
                        if not disambigmatch_2:
                            assert False, u'{}: this is a trap to test ReTag_wikinotedisambigpage_2. please check it! \n{}'.format(get_cur_info(), t2)
                    if not disambigmatch:#从我设计的本意上说，当disambigmatch为空时，disambigmatch_2按说也为空。如果违反，则报错
                        if disambigmatch_2:
                            assert False, u'{}: this is a trap to test ReTag_wikinotedisambigpage_2. please check it! \n{}'.format(get_cur_info(), t2)
                    disambigmatch = disambigmatch_2
                    # -----
                    if disambigmatch:#means there is disambiguation page link,such as <a href="/wiki/Hypothesis_(disambiguation)" class="mw-disambig" title="Hypothesis (disambiguation)">Hypothesis (disambiguation)</a>
                        t3 = disambigmatch.group(1)
                        bFlagAmbiguous = True
                        hatnotelinkdisambig.append(t3)
                        # --------------------------之间的这段代码是为了使消歧页的候选概念更精确一些。包含头部的那些信息，并且去掉see also中的概念。参看词语Frame的例子。也有不成功的时候，比如post，此时只过滤see also中的概念
                        list_disambigerror_options = GetWikiAmbiguousTitles(t3)
                        url_disambigWikipage = u'https://en.wikipedia.org/wiki/{}'.format(t3)
                        (list_DisambigItems, list_SeealsoItems) = collectItemOnDisambigWikiPage(url_disambigWikipage)
                        ultimate_list_DisambigOptions = []
                        if len(list_DisambigItems) > 1:
                            # 如果自行在消歧页中查找成功了，则优先用自己的（比较全，而且已过滤了see also概念）
                            ultimate_list_DisambigOptions = list_DisambigItems
                        else:
                            # 如果自行在消歧页中查找失败了
                            if len(list_SeealsoItems) == 0:
                                # 如果自行在消歧页中查找失败了，且see also也失败了，则只能使用wikipedia api自动返回的了
                                ultimate_list_DisambigOptions = list_disambigerror_options
                            else:
                                # 如果自行在消歧页中查找失败了，但see also查找成功了，则使用把list_disambigerror_options中的属于see also的滤除掉
                                for item in list_disambigerror_options:
                                    if item not in list_SeealsoItems:
                                        ultimate_list_DisambigOptions.append(item)
                                    else:
                                        # 因为disambigerror_options都是按页面出现顺序来的，只要发现一个出现在see also中的，就意味着后面的都属于see also，就都不用再看了。
                                        # 也是为了避免诸如词语post的处理不完善的问题。
                                        break
                        # -------------------
                        for tt in ultimate_list_DisambigOptions:
                        #for tt in GetWikiAmbiguousTitles(t3):
                            #print tt   #only test
                            if tt.lower() == dictret[u'Title'].lower():# if tt is same with Title, abandon it
                                continue
                            if ReTag_disambigchar.search(tt) == None and ReTag_listofchar.match(tt) == None:# if tt is a disambiguation title or List of xxx, abandon it
                                if lookforfinalpage:
                                    ttfinal = GetFinalTitleinWiki(tt)
                                else:
                                    ttfinal = tt
                                if ttfinal and ReTag_disambigchar.search(ttfinal) == None and ReTag_listofchar.match(ttfinal) == None:
                                    hatnotelinkdisambigitems.append(ttfinal)
                    if bFlagAmbiguous == False:
                        othernotematch = ReTag_wikinoteotherpage.search(t2)
                        if othernotematch:#means there is a non-disambiguation page link, such as <a href="/wiki/Theorem" title="Theorem">Theorem</a>
                            t3 = othernotematch.group(1)
                            if lookforfinalpage:
                                t3final = GetFinalTitleinWiki(t3)
                            else:
                                t3final = t3
                            if t3final and ReTag_disambigchar.search(t3final) == None and ReTag_listofchar.match(t3final) == None:
                                hatnotelinks.append(t3final)


            if len(hatnotelinks)!=0:
                dictret[u'HatnoteLinkItems'] = hatnotelinks
            if len(hatnotelinkdisambig)!=0:
                dictret[u'HatnoteLinkDisambig'] = hatnotelinkdisambig
            if len(hatnotelinkdisambigitems)!=0:
                dictret[u'HatnoteLinkDisambigItems'] = hatnotelinkdisambigitems

            break
        except wikipedia.exceptions.DisambiguationError as disambigerror:
            dictret[u'bNormal'] = False
            dictret[u'bDisambig'] = True
            tmpset = []
            #--------------------------之间的这段代码是为了使消歧页的候选概念更精确一些。包含头部的那些信息，并且去掉see also中的概念。参看词语Frame的例子。也有不成功的时候，比如post，此时只过滤see also中的概念
            list_disambigerror_options = disambigerror.options
            url_disambigWikipage = u'https://en.wikipedia.org/wiki/{}'.format(text)
            (list_DisambigItems, list_SeealsoItems) = collectItemOnDisambigWikiPage(url_disambigWikipage)
            ultimate_list_DisambigOptions = []
            if len(list_DisambigItems)>1:
                #如果自行在消歧页中查找成功了，则优先用自己的（比较全，而且已过滤了see also概念）
                ultimate_list_DisambigOptions = list_DisambigItems
            else:
                #如果自行在消歧页中查找失败了
                if len(list_SeealsoItems)==0:
                    #如果自行在消歧页中查找失败了，且see also也失败了，则只能使用wikipedia api自动返回的了
                    ultimate_list_DisambigOptions = list_disambigerror_options
                else:
                    #如果自行在消歧页中查找失败了，但see also查找成功了，则使用把list_disambigerror_options中的属于see also的滤除掉
                    for item in list_disambigerror_options:
                        if item not in list_SeealsoItems:
                            ultimate_list_DisambigOptions.append(item)
                        else:
                            #因为disambigerror_options都是按页面出现顺序来的，只要发现一个出现在see also中的，就意味着后面的都属于see also，就都不用再看了。
                            # 也是为了避免诸如词语post的处理不完善的问题。
                            break
            #-------------------
            #for tt in disambigerror.options:
            for tt in ultimate_list_DisambigOptions:
                if tt.lower() != dictret[u'Title'].lower():
                    if ReTag_disambigchar.search(tt) == None and ReTag_listofchar.match(tt) == None:  # if tt is a disambiguation title or List of xxx, abandon it
                        if lookforfinalpage:
                            ttfinal = GetFinalTitleinWiki(tt)
                        else:
                            ttfinal = tt
                        if ttfinal and ReTag_disambigchar.search(ttfinal) == None and ReTag_listofchar.match(ttfinal) == None:
                            tmpset.append(ttfinal)
            dictret[u'DisambigItems'] = tmpset # record the referred ambiguous page titles
            break
        except wikipedia.exceptions.RedirectError as redirecterror:
            dictret[u'bNormal'] = False
            dictret[u'bRedirect'] = True

            try:
                redirect = wikipedia.page(text,auto_suggest=False, redirect=True)
                dictret[u'RedirectTitle'] = redirect.title
                if permitRedirectQuery:
                    requery = WikiInfo(dictret[u'RedirectTitle'],retrytimes=1,permitRedirectQuery=False) # call itself to get the info of redirect page
                    #occupy the position of original title , to save the information of redirect title
                    #dictret[u'Title']
                    dictret[u'bDisambig'] = requery['bDisambig']
                    dictret[u'DisambigItems'] = requery['DisambigItems']
                    #dictret[u'bRedirect'] = None  # is there a redirect action for current title
                    #dictret[u'RedirectTitle'] = None  # the redirect title
                    dictret[u'bPageErr'] = requery['bPageErr']  # is there a page error
                    dictret[u'bHttpTimeOut'] = requery['bHttpTimeOut']  # is there a Internet network problem
                    dictret[u'HatnoteLinkDisambig'] = requery['HatnoteLinkDisambig']  # if there is a ambigous link in HatNote of the wikipage, such as 'hypothesis'
                    dictret[u'HatnoteLinkDisambigItems'] = requery['HatnoteLinkDisambigItems']  # if there is a ambigous link in HatNote of the wikipage, such as 'hypothesis', get its detailed referred items.
                    dictret[u'HatnoteLinkItems'] = requery['HatnoteLinkItems']  # if there is some non-ambigous links in HatNote of the wikipage, such as 'hypothesis'
                break
            except wikipedia.exceptions.DisambiguationError as disambigerror2:#if the redirect page is a disambiguation page, need the following code
                #eg: Rocket ship (disambiguation)
                dictret[u'RedirectTitle'] = disambigerror2.title
                if permitRedirectQuery:
                    requery = WikiInfo(dictret[u'RedirectTitle'],retrytimes=1,permitRedirectQuery=False) # call itself to get the info of redirect page
                    #occupy the position of original title , to save the information of redirect title
                    #dictret[u'Title']
                    dictret[u'bDisambig'] = requery['bDisambig']
                    dictret[u'DisambigItems'] = requery['DisambigItems']
                    #dictret[u'bRedirect'] = None  # is there a redirect action for current title
                    #dictret[u'RedirectTitle'] = None  # the redirect title
                    dictret[u'bPageErr'] = requery['bPageErr']  # is there a page error
                    dictret[u'bHttpTimeOut'] = requery['bHttpTimeOut']  # is there a Internet network problem
                    dictret[u'HatnoteLinkDisambig'] = requery['HatnoteLinkDisambig']  # if there is a ambigous link in HatNote of the wikipage, such as 'hypothesis'
                    dictret[u'HatnoteLinkDisambigItems'] = requery['HatnoteLinkDisambigItems']  # if there is a ambigous link in HatNote of the wikipage, such as 'hypothesis', get its detailed referred items.
                    dictret[u'HatnoteLinkItems'] = requery['HatnoteLinkItems']  # if there is some non-ambigous links in HatNote of the wikipage, such as 'hypothesis'
                break

            except Exception as inst:
                # print type(inst)
                # print inst.args
                # print inst
                note = '!!!!there is a exception:  requery = wikiinfo(dictret[\'RedirectText\'],retrytimes=1,permitRedirectQuery=False) # call itself'
                dictret[u'Exception'] = 'text:{} ; type(inst):{} ; inst.args:{} ; inst:{} ;note:{}'.format(text, type(inst), inst.args, inst, note)
                break
        except wikipedia.exceptions.PageError:
            dictret[u'bNormal'] = False
            dictret[u'bPageErr'] = True
            break
        except wikipedia.exceptions.HTTPTimeoutError:
            dictret[u'bNormal'] = False
            dictret[u'bHttpTimeOut'] = True
            continue
        except Exception as inst:
            # print type(inst)
            # print inst.args
            # print inst
            dictret[u'Exception'] = 'text:{} ; type(inst):{} ; inst.args:{} ; inst:{} '.format(text, type(inst), inst.args, inst)
            break

    return dictret

def GetFinalTitleinWiki(title,redirct=False):
    '''
    for a title, to find its final title in wikipedia.
    if there is a disambiguation page, pageerror, httperro, return None
    :param title:
    :return:
    '''
    try:
        wikiret = wikipedia.page(title,auto_suggest=False,redirect=redirct)
        return wikiret.title
    except wikipedia.exceptions.DisambiguationError as disambigerror:
        #print title+" is a disambiguation page, return None"
        return None
    except wikipedia.exceptions.RedirectError as redirecterror:
        #recursively call itself to get final title in wiki
        t = GetFinalTitleinWiki(redirecterror.title,True)
        return t
    except wikipedia.exceptions.PageError:
        #print title+" is not a wikipedia keyword, return None"
        return None
    except wikipedia.exceptions.HTTPTimeoutError:
        #print title+" Network Error! HttpTimeout, return None"
        return None
    except Exception as inst:
        print(type(inst))
        print inst.args
        print inst.message
        print inst
        print "Title: "+title.encode('utf-8')+ " , induces a Exception! (GetFinalTitleinWiki(title,redirct=False))"
        return None
    return None


def collectSeealsoItemsFromDisambigWikipage(divcontent_text):
    # 取一个wiki消歧页中的see also中的概念信息（这个方法不太精确，如果页面显示的文本 与 实质链接不一致，这个方法只是返回页面显示的文本）
    ret = []
    '''
    for i in range(0,10):
        try:
            response = requests.get(url)
            break
        except:
            if i==9:
                return ret
            else:
                time.sleep(10)
                continue
    if response.status_code > 400 :#意味着the status code of the response is between 400 and 600 to see if there was a client error or a server error.
        return ret

    soup = BeautifulSoup(response.content.decode('utf-8'))

    # 找到核心的页面内容
    divcontent = soup.find('div', id='mw-content-text')

    if divcontent == None:
        assert False, u'"{}" fails to find tag "div id=mw-content-text"'.format(url)


    text = divcontent.get_text()
    '''
    text = divcontent_text
    seealsobeginpos = text.rfind(u'See also[edit]') + len(u'See also[edit]')
    seealsoendpos = text.rfind(u'This disambiguation page lists articles associated with')
    if seealsobeginpos == (-1+len(u'See also[edit]'))  or seealsoendpos == -1: #Hypothesis (disambiguation)就没有see also这一块文本
        #assert False
        return ret
    else:
        seealsoarea = text[seealsobeginpos:seealsoendpos]
        # print '------'
        # print seealsoarea
        # print '----------'
        for t in seealsoarea.split(u'\n'):
            t = t.strip()
            if len(t) > 0:
                ret.append(t)
        return ret


def collectItemOnDisambigWikiPage(url):
    '''
    1)ret_DisambigItems 取wiki消歧页中的概念。取得比wiki api返回的要全面，比如Association一词；同时，把see also后的概念给剔除了。
    代码适用性不算强，比如post，会提取失败。对于失败的，返回长度为0的空链表。对于成功的，返回 概念链表。
    2)ret_SeealsoItems  取wiki消歧页中See also的概念
    :param url: 
    :return: 
    '''
    # url = 'https://en.wikipedia.org/wiki/Association'
    # url = 'https://en.wikipedia.org/wiki/Stimulation_(disambiguation)'
    # url = 'https://en.wikipedia.org/wiki/MS'
    # url = 'https://en.wikipedia.org/wiki/post'#这个提取失败
    # url = 'https://en.wikipedia.org/wiki/Wizard'
    # url = 'https://en.wikipedia.org/wiki/pointer'

    ret_DisambigItems = []
    ret_SeealsoItems = []
    for i in range(0, 10):
        try:
            response = requests.get(url)
            break
        except:
            if i == 9:
                return (ret_DisambigItems, ret_SeealsoItems)
            else:
                time.sleep(10)
                continue
    if response.status_code > 400:  # 意味着the status code of the response is between 400 and 600 to see if there was a client error or a server error.
        return (ret_DisambigItems, ret_SeealsoItems)

    soup = BeautifulSoup(response.content.decode('utf-8'), "html.parser")

    # 找到核心的页面内容
    divcontent = soup.find('div', id='mw-content-text')

    if divcontent == None:
        assert False, u'{}: "{}" fails to find tag "div id=mw-content-text"'.format(get_cur_info(), url)

    # 1) 取wiki消歧页中的概念 ret_DisambigItems
    for t in divcontent.children:
        #print t
        if isinstance(t, NavigableString):  # 只保留bs4.element.Tag类型的孩子
            continue
        if t.find(u'span', id=u"See_also") != None:  # See also后的丢掉，不要了。严格来讲，不属于 可能的概念词义
            break
        if t.find(u'span', id=u"References") != None: # References后面的也不要了
            break
        if t.name == 'table':  # eg: Association
            continue
        if t.name == 'h2':  # eg: Stimulation (disambiguation)
            continue
        if t.name == 'h3':  # eg: MS
            continue
        if t.name == 'h4' or t.name == 'h5' or t.name == 'h6':# eg. Love (disambiguation)
            continue
        if t.name == 'div':  # eg: MS
            continue
        if t.name == 'dl':# eg: Helmet (disambiguation)
            continue
        if t.find('a') == None:  # 不含超级链接标记直接删掉
            continue
        # print 'divchildren:', t

        if t.name == 'p' or t.name == 'ul':
            # 对于段落，往往是开头那一块，如Frame的开关部分。直接提取全部链接。
            if t.name == 'p':
                titlelinks = t.find_all(u'a')  # 找所有的超级链接标记
                # print titlelink
                for titlelink in titlelinks:
                    if titlelink != None:
                        # print titlelink.find(text=u'href="/wiki/')
                        if titlelink.attrs.has_key(u'href') and titlelink.attrs.has_key(u'title'):
                            if titlelink.attrs[u'href'].find('/wiki/') != -1:
                                # print titlelink.attrs[u'title']
                                ret_DisambigItems.append(titlelink.attrs[u'title'])
            # 对于列表ul，往往是下方的内容，如Frame的下方，分析其孩子li，每行只提取一个链接。避免Frames (Lee DeWyze album), a 2013 album by Lee DeWyze
            if t.name == 'ul':
                for tt in t.children:
                    if isinstance(tt, NavigableString):  # 只保留bs4.element.Tag类型的孩子
                        continue
                    if tt.find('a') == None:  # 不含超级链接标记直接删掉
                        continue
                    titlelink = tt.find(u'a')  # 每行只找第一个超级链接标记，避免一些误差
                    if titlelink != None:
                        if titlelink.attrs.has_key(u'href') and titlelink.attrs.has_key(u'title'):
                            if titlelink.attrs[u'href'].find('/wiki/') != -1:
                                # print titlelink.attrs[u'title']
                                ret_DisambigItems.append(titlelink.attrs[u'title'])
                    #----
                    listofsecond_li = tt.find_all(u'li')
                    for eachli in listofsecond_li:
                        titlelink = eachli.find(u'a')
                        if titlelink != None:
                            if titlelink.attrs.has_key(u'href') and titlelink.attrs.has_key(u'title'):
                                if titlelink.attrs[u'href'].find('/wiki/') != -1:
                                    if titlelink.attrs[u'title'] not in ret_DisambigItems:
                                        ret_DisambigItems.append(titlelink.attrs[u'title'])

                        mm = findwikiconceptLinkinDisambigWikiPage_litag(eachli)
                        if len(mm) > 0:
                            for item in mm:
                                if item not in ret_DisambigItems:
                                    ret_DisambigItems.append(item)
                    #----
                    #if tt.name == 'li':  # 看看子标记中，是不是仍有下级链接
                    #    mm = findwikiconceptLinkinDisambigWikiPage_litag(tt)
                    #    if len(mm) > 0:
                    #        # print mm
                    #        ret_DisambigItems.extend(mm)
        else:
            assert False, u'{}: "{}" exist some unhandled tag "{}"! plase check it!'.format(get_cur_info(), url, t.name)

    # 2) 取wiki消歧页中See also的概念 ret_SeealsoItems
    divcontent_text = divcontent.get_text()
    ret_SeealsoItems = collectSeealsoItemsFromDisambigWikipage(divcontent_text)

    return (ret_DisambigItems, ret_SeealsoItems)


def findwikiconceptLinkinDisambigWikiPage_litag(father_litag):
    # 如词语Frame中，有些列表会有子选项，如frame的Bicycle frame，这个函数专门用于提取子选项motorcycle frame
    assert father_litag.name == 'li'
    ret = []
    for child in father_litag.children:
        if isinstance(child, NavigableString):  # 只保留bs4.element.Tag类型的孩子
            continue
        if child.find('a') == None:  # 不含超级链接标记直接删掉
            continue
        titlelink = child.find(u'a')  # 每行只找第一个超级链接标记，避免一些误差
        if titlelink.attrs.has_key(u'href') and titlelink.attrs.has_key(u'title'):
            if titlelink.attrs[u'href'].find('/wiki/') != -1:
                #print titlelink.attrs[u'title']
                ret.append(titlelink.attrs[u'title'])
    return ret





version = '1.0'
def main():
    print "this is main begin ..."

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=__doc__)
    parser.add_argument("input",
                        help="the inputfile that contains the word pairs. each line is a word pair, that is two short text splitted with a tab")

    groupO = parser.add_argument_group('Outputfile')
    groupO.add_argument("-o", "--output",
                        help="the outputfile")


    groupS = parser.add_argument_group('Special')
    groupS.add_argument("-v", "--version", action="version",
                        version='%(prog)s ' + version,
                        help="print program version")

    args = parser.parse_args()


    inputfilepath = args.input
    outfilepath = args.output


    infile = fileinput.FileInput(inputfilepath)
    #outfile = open(outfilepath, 'w')  # w : open a new file
    if os.path.isfile(outfilepath):
        echo = raw_input(u'{}: {} has existed. \nDo you want to append output to the file?\nInput(y,n):'.format(get_cur_info(), outfilepath))
        time.sleep(1)
        if echo.lower() != 'y':
            print 'please change a filename for output file'
            return

    outfile = open(outfilepath, 'a')  # a : append text to a existing file

    wordpairlist = []
    for line in lines_from(infile):
        wordpair = line.split("\t")
        if len(wordpair) != 2:
            print 'the formation of input file is not "word1\\tword2"'
            break
        wordpairlist.append(wordpair)
    print len(wordpairlist)


    count = 0
    LookedWordDict = {}
    for wordpair in wordpairlist:
        count += 1
        if count <= 0: #when a interrupt , to continue
            continue
        print "Pair No.{}/{}  {}".format(count, len(wordpairlist), wordpair)
        print time.strftime("%Y-%m-%d %X",time.localtime())
        for word in wordpair:
            #word = "Stimulation"
            #word = "Frame"
            #word = "post"
            #word = "AI"
            #for i in range(0,20):
            #    print word

            if LookedWordDict.has_key(word):
                line = LookedWordDict.get(word)
            else:
                line = FormatOutputofWikiInfo(WikiInfo(word))
                line = (line+'\n').encode('utf-8')
                LookedWordDict[word]=line
            print line
            outfile.write(line)

    outfile.close()
    infile.close()

    '''
    print '--------------------------'
    print 'test self-driving car  , redirect page'
    print FormatOutputofWikiInfo(WikiInfo(u'self-driving car'))

    print '--------------------------'
    print 'test Rocket ship (disambiguation)  , redirect to a disambiguation page'
    print FormatOutputofWikiInfo(WikiInfo(u'Rocket ship (disambiguation)'))

    print '--------------------------'
    print 'test Joule  , normal page with a disambiguate hatnote'
    print FormatOutputofWikiInfo(WikiInfo(u'Joule'))

    print '--------------------------'
    print 'test fault  , disambiguate page'
    print FormatOutputofWikiInfo(WikiInfo(u'fault'))

    print '--------------------------'
    print 'test Spacecraft  , normal page with two disambiguation hatnote'
    print FormatOutputofWikiInfo(WikiInfo(u'Spacecraft'))

    print '--------------------------'
    print 'test Darwin  , disambiguation page'
    print FormatOutputofWikiInfo(WikiInfo(u'Darwin'))

    print WikiInfo(u"Spacecraft")
    print WikiInfo(u"hypothesis")
    print WikiInfo(u"Legion of Honour")
    print WikiInfo(u'Legion of Honor')
    '''







    '''
    wiki.set_lang("en")

    word = "skdfjlskf sdkfslfjs sadkflj"
    print word
    print "wiki.search():   ",
    ret = wiki.search(word)
    print len(ret)
    print ret

    print '--------------------------------'
    print "wiki.search(word,results=20,suggestion=True):   ",
    ret = wiki.search(word,results=20,suggestion=True)
    print len(ret)
    print ret


    print '--------------------------------'
    print "wiki.page():    "
    ret = wiki.page(word)
    print "categories:   ", ret.categories
    print "summary:  " + ret.summary
    # print "html():  "+ret.html()

    print '--------------------------------'
    print "wiki.summary():   ",
    print wiki.summary(word)


    print '--------------------------------'
    print "wiki.summary(word,auto_suggest=False,redirect=False):   ",
    print wiki.summary(word,auto_suggest=False,redirect=False)
    print '--------------------------------'
    print "wiki.page(word,auto_suggest=False,redirect=False):    "
    ret = wiki.page(word,auto_suggest=False,redirect=False)
    print "summary:  " + ret.summary
    '''





    '''
    print wiki.search("Barack")
    print wiki.suggest("Baark Obama")


    print wiki.search("Ford",results=3)
    print wiki.summary("GitHub",sentences=1)

    try:
        mercury = wiki.summary("Mercury")
    except wiki.exceptions.DisambiguationError as e:
        print e.options

    print wiki.summary("zvv")

    ny = wiki.page("New York")
    print ny.title
    print ny.url
    print ny.content
    print ny.images[0]
    print ny.links[0]


    wiki.set_lang("en")
    print wiki.summary("Francois Hollande")

    aa = wiki.WikipediaPage(title="China")
    print aa.summary
    print aa.links
    print aa.categories
    print aa.content

    print 'en' in wiki.languages()
    print wiki.languages()['es']
    '''






    print "this is main end!"





if __name__ == '__main__':
    print WikiInfo('hello')