-- 현재 접속한 db와 사용할 db가 다른 상태

use mldb3

SELECT * from orderlist o 

select o.orderid '주문아이디', p.id as '주문물건번호'
from orderlist o 
inner join product p 
on o.productid = p.id 


-- 어떤 회원이 어떤 물건을 얼마나 구매했는가?(회원id, 물건id, 총구매가격, 주문id, 회원name)
select m.id , o2.productid , o2.totalprice , o2.orderid , m.name 
from orderlist o2
inner join `member` m 
on o2.memberid = m.id 

-- park회원이 어떤 물건을 얼마나 구매했는가?
-- (주문id, 회원name, 물건id, 총구매가격)
-- 물건id 역순 정렬 
select o2.orderid, m.name , o2.productid , o2.totalprice - 1000
from orderlist o2
inner join `member` m 
on o2.memberid = m.id and m.id = 'park'
order by productid desc



select user, host from mysql.user;


select o2.orderid, m.name , o2.productid , o2.totalprice - 1000
from orderlist o2
inner join `member` m 
on o2.memberid = m.id and m.id like 'a%'
order by productid desc

select o2.orderid, m.name , o2.productid , o2.totalprice - 1000
from orderlist o2
inner join `member` m 
on o2.memberid = m.id and m.id not like '%a%' -- % 0~n
order by productid desc

select o2.orderid, m.name , o2.productid , o2.totalprice - 1000
from orderlist o2
inner join `member` m 
on o2.memberid = m.id and m.id  like 'a____' -- _ => 1
order by productid desc


select o2.memberid , COUNT(*), SUM(totalprice) 
from orderlist o2
inner join `member` m 
on o2.memberid = m.id and m.id = 'apple'

select o3.memberid , COUNT(*), SUM(totalprice) 
from orderlist o3
inner join `member` m2 
on o3.memberid = m2.id 
GROUP by m2.id

-- 개인별 몇개를, 최대 얼마만큼 구매했을까요?
-- >join
select m.id, COUNT(*), MAX(totalprice) 
from orderlist o2
inner join `member` m 
on o2.memberid = m.id 
GROUP by m.id 


-- 물건별 몇 개를 얼마만큼 구매했을까요?
-- > orderlist table에서 그룹별로 검색
select count(*), sum(totalprice) 
FROM orderlist o5
group by o5.productid

-- 물건별 몇 개를 얼마만큼 구매했고, 최대 구매금액은 얼마인가요?
-- > orderlist table에서 그룹별로 검색 
select productid, sum(totalprice), MAX(totalprice) 
FROM orderlist o5
group by o5.productid

-- 주문이 된 상품들 목록을 가지고 오고 싶을 때
select DISTINCT productid from orderlist o 

select o6.productid, p6.name, COUNT(*) 
from orderlist o6
inner join product p6
on o6.productid = p6.id 
GROUP by o6.productid


-- member, board

-- 1) 개인별 게시판 글 작성수
SELECT writer, COUNT(*) 
FROM board b
group by b.writer 

-- 2) 개인별 게시판 글 작성수, 개인id, 개인name, 개인tel
SELECT writer as userid, count(*), m.id , m.name , m.tel 
from board b
inner join `member` m 
on b.writer = m.id 
group by b.writer 

-- 3) 게시판 제목, 내용, 작성자이름, 작성자tel
SELECT title as 제목 , content as 내용 , writer , m.tel 
FROM board b
inner join `member` m
on b.writer = m.id

-- 4) park이 작성한 글 작성수, 최근 게시물 번호
SELECT b.writer as 작성자, count(*) as 글작성수, max(b.id) as 최근게시물번호
FROM board b 
join `member` m 
on b.writer = m.id and m.id = 'park' 

-- 5) apple이 작성한 글 중 java가 들어간 게시물 갯수
select count(*)
from board b
where writer = 'apple' and content like '%java'

-- 6) park이 작성한 글 중 제일 오래된 게시물 번호, 제목, 내용


-- 7) 게시판 제목에 어떤 제목들이 올라왔나? (중복제거)
select DISTINCT title from board b 


-- 8) 게시판 내용 중 jsp가 들어간 글의 작성자이름과 작성자연락처 
select m.name , m.tel , b.content 
FROM board b 
inner join `member` m 
on m.id = b.writer 
where content like '%jsp'


-- subquery
select DISTINCT o.productid 
	from orderlist o 
	where o.memberid = 'park'
	
SELECT * FROM product p 
WHERE p.id in (
	select DISTINCT o.productid 
	from orderlist o 
	where o.memberid = 'park'	
)














