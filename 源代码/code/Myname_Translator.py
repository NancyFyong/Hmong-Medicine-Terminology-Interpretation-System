import pymysql
from PIL import Image
def login_database():
  host = 'localhost'
  username = 'root'
  password = '12345678'
  db_name = 'myy_system'
  charset = 'utf8'
  db=pymysql.connect(
      host=host,
      user=username,
      password=password,
      db=db_name,
      charset=charset
  )
  cursor = db.cursor()
  return db,cursor
def translator_func(text):
  db,cursor=login_database()
  sql = "select * from cjcy where name = %s or mw_name like %s"
  try:
      if text == '':
        return None,"输入为空，请重新输入"
      my_labels = ( '名称', '苗药名', '俗名', '来源', '性味', '功效')
      cursor.execute(sql,(text,'%'+text+'%'))
      result = cursor.fetchall()  # 返回所有数据
      print(result)
      all_result=''
      for item in result:
        image = item[-1]
        result_string = "\n".join([f"{label}: {value}" for label, value in zip(my_labels, item[1:])])
        all_result += result_string+'\n'
      image = image if image != '' else None
      all_result = all_result if all_result != '' else '输入有误'
      print(all_result)
      return image,all_result
  except Exception as e:
      db.rollback()
      print(e)
  finally:
      cursor.close()
      db.close()

translator_func('kuad bed vud')
