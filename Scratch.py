import MainMod as M


#M.ProcessData()

speed = 10 ###slow = 10, fast 25
states = ['FL', 'GA', 'TX', 'US']
start_date = '2017-8-1'
end_date = '2018-8-1'

for ST in states:
    [a, b] = M.MkRunner(ST, start_date, end_date, 'abs')
    M.MkMap(ST, start_date, end_date)
    M.AddRunner2Map(ST, 0, int(a), 3000, start_date)
    M.MkVideo(ST, start_date, end_date,speed)


# print('State: ' + ST)
# print('start_date: ' + start_date)
# print('end_date: ' + end_date)
#
# [a,b] = MkRunner(ST, start_date, end_date, 'abs')
# MkMap(ST,start_date, end_date)
# AddRunner2Map(ST,0, int(a), 3000, start_date)