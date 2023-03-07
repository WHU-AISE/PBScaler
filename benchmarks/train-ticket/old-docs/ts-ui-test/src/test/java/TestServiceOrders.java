import org.openqa.selenium.Alert;
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


public class TestServiceOrders {
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
    public void testOrders()throws Exception{
        driver.get(baseUrl + "/");
        WebElement elementRefreshOrdersBtn = driver.findElement(By.id("refresh_order_button"));
        WebElement elementOrdertypeGTCJ = driver.findElement(By.xpath("//*[@id='microservices']/div[4]/div[1]/h3/input[1]"));
        WebElement elementOrdertypePT = driver.findElement(By.xpath("//*[@id='microservices']/div[4]/div[1]/h3/input[2]"));
        elementOrdertypeGTCJ.click();
        elementOrdertypePT.click();
        if(elementOrdertypeGTCJ.isEnabled() || elementOrdertypePT.isEnabled()){
            elementRefreshOrdersBtn.click();
            System.out.println("Show Orders according database!");
        }
        else {
            elementRefreshOrdersBtn.click();
            Alert javascriptConfirm = driver.switchTo().alert();
            javascriptConfirm.accept();
            elementOrdertypeGTCJ.click();
            elementOrdertypePT.click();
            elementRefreshOrdersBtn.click();
        }
        List<WebElement> ordersList = driver.findElements(By.xpath("//table[@id='all_order_table']/tbody/tr"));
        if (ordersList.size() > 0) {
            System.out.printf("Success,Orders List's size is %d.%n", ordersList.size());
        } else
            System.out.println("False,Security Config List's size is 0 or Failed");
        Assert.assertEquals(ordersList.size() > 0, true);

    }
    @AfterClass
    public void tearDown() throws Exception {
        driver.quit();
    }
}
