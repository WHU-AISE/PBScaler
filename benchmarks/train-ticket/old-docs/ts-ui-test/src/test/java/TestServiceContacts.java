import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.util.List;
import java.util.concurrent.TimeUnit;


public class TestServiceContacts {
    private WebDriver driver;
    private String baseUrl;
    @BeforeClass
    public void setUp() throws Exception {
        System.setProperty("webdriver.chrome.driver", "D:/Program/chromedriver_win32/chromedriver.exe");
        driver = new ChromeDriver();
        baseUrl = "http://10.141.212.24/";
        driver.manage().timeouts().implicitlyWait(30, TimeUnit.SECONDS);
    }
    @Test
    public void testContacts()throws Exception{
        driver.get(baseUrl + "/");
        driver.findElement(By.id("refresh_contacts_button")).click();
        Thread.sleep(1000);
        List<WebElement> contactsList = driver.findElements(By.xpath("//table[@id='contacts_list_table']/tbody/tr"));
        //List<WebElement> contactsList = driver.findElements(By.xpath("//table[@id='contacts_booking_list_table']/tbody/tr"));
        if(contactsList.size() > 0) {
            System.out.printf("Success,Contacts List's size is %d.%n", contactsList.size());
        }
        else
            System.out.println("False,Contacts List's size is 0 or Failed");
        Assert.assertEquals(contactsList.size() > 0,true);
    }
    @AfterClass
    public void tearDown() throws Exception {
        driver.quit();
    }
}
