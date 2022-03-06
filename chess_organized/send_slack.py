#! python3

from slack_sdk import WebClient

def send_msg(msg):
    client = WebClient(token="xoxp-1140170931975-1161092683524-1178761298416-2d25f55c0faa0a460202282ea07a3c63")
    client.chat_postMessage(channel='#notification', text=msg)

