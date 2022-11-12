import asyncio

from swin_td_bu_att.image_demo import parse_args, async_main, main

args = parse_args()
if args.async_test:
	asyncio.run(async_main(args))
else:
	main(args)
